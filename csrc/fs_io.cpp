// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <Python.h>

#include <errno.h>
#include <fcntl.h>
#include <sys/xattr.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(__ARM_FEATURE_CRC32)
  #include <arm_acle.h>
#endif

#if defined(O_DIRECT)
constexpr int kODirectFlag = O_DIRECT;
#else
constexpr int kODirectFlag = 0;
#endif

extern "C" {

namespace {

// The CRC32 of a stored block, recorded by fs/io.py write_checksums as a
// 4-byte big-endian extended attribute.
constexpr const char* kCrcXattr = "user.vllm_kv_crc32";

// CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320): the same checksum as
// Python's zlib.crc32. Slice-by-8 tables, or the ARMv8 CRC32 instructions
// when the build enables them (-march=armv8-a+crc).
struct Crc32Tables {
  uint32_t t[8][256];
  Crc32Tables() {
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (int k = 0; k < 8; k++) {
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      t[0][i] = c;
    }
    for (uint32_t i = 0; i < 256; i++) {
      for (int j = 1; j < 8; j++) {
        t[j][i] = (t[j - 1][i] >> 8) ^ t[0][t[j - 1][i] & 0xFF];
      }
    }
  }
};

inline uint32_t crc32_of(const unsigned char* p, size_t n) {
  uint32_t crc = 0xFFFFFFFFu;
#if defined(__ARM_FEATURE_CRC32)
  while (n >= 8) {
    uint64_t v;
    memcpy(&v, p, 8);
    crc = __crc32d(crc, v);
    p += 8;
    n -= 8;
  }
  while (n--) {
    crc = __crc32b(crc, *p++);
  }
#else
  static const Crc32Tables tables;
  const auto& t = tables.t;
  #if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
  while (n >= 8) {
    uint32_t one, two;
    memcpy(&one, p, 4);
    memcpy(&two, p + 4, 4);
    one ^= crc;
    crc = t[7][one & 0xFF] ^ t[6][(one >> 8) & 0xFF] ^
          t[5][(one >> 16) & 0xFF] ^ t[4][one >> 24] ^ t[3][two & 0xFF] ^
          t[2][(two >> 8) & 0xFF] ^ t[1][(two >> 16) & 0xFF] ^ t[0][two >> 24];
    p += 8;
    n -= 8;
  }
  #endif
  while (n--) {
    crc = t[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);
  }
#endif
  return ~crc;
}

// Reads the recorded CRC32 of `path`. Returns 1 with the value in `out`, 0
// when the file has no recorded checksum (or xattrs are unsupported), and -1
// when the attribute exists but is malformed.
inline int read_recorded_crc(const char* path, uint32_t* out) {
  unsigned char raw[4];
#if defined(__APPLE__)
  const ssize_t got = getxattr(path, kCrcXattr, raw, sizeof(raw), 0, 0);
#else
  const ssize_t got = getxattr(path, kCrcXattr, raw, sizeof(raw));
#endif
  if (got < 0) {
    return 0;
  }
  if (got != sizeof(raw)) {
    return -1;
  }
  *out = (uint32_t{raw[0]} << 24) | (uint32_t{raw[1]} << 16) |
         (uint32_t{raw[2]} << 8) | uint32_t{raw[3]};
  return 1;
}

// Returns 0 on success, or the std::error_code's POSIX-compatible value on
// failure, mirroring the errno convention used by the syscalls below.
inline int ensure_parent_dirs(const std::string& path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (parent.empty()) {
    return 0;
  }
  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  return ec ? ec.value() : 0;
}

// Core single-block store: src/size are raw pointer + byte count. Returns 0
// on success, or the errno of the failing step on failure -- captured
// before any subsequent cleanup call can overwrite it. On failure, the temp
// file is removed.
inline int _store_block(const char* tmp_path, const char* dest_path,
                        const char* src, size_t size, bool use_o_direct) {
  if (access(dest_path, F_OK) == 0) {
    return 0;  // Already present.
  }

  if (const int err = ensure_parent_dirs(dest_path); err != 0) {
    return err;
  }

  const int o_direct_flag = use_o_direct ? kODirectFlag : 0;
  const int fd = open(
      tmp_path, O_CREAT | O_EXCL | O_WRONLY | O_TRUNC | o_direct_flag, 0644);
  if (fd < 0) {
    return errno;
  }

  const ssize_t written = write(fd, src, size);
  if (written < 0 || static_cast<size_t>(written) != size) {
    const int err = written < 0 ? errno : EIO;
    close(fd);  // Best-effort cleanup; the real error is already captured.
    unlink(tmp_path);
    return err;
  }

  if (close(fd) != 0) {
    const int err = errno;
    unlink(tmp_path);
    return err;
  }

  if (rename(tmp_path, dest_path) != 0) {
    const int err = errno;
    unlink(tmp_path);
    return err;
  }

  return 0;
}

// Core single-block load: dst/size are raw pointer + byte count. Returns 0
// on success, or the errno of the failing step on failure. Removes the source
// file ONLY on a provable short read (the read completed but returned fewer
// bytes than requested): stores are atomic, so a too-short file is genuine
// corruption. Open failures and read errors (bytes_read < 0) are
// transient/ambiguous and leave the file untouched; a close failure after a
// full read is harmless and does not fail the load.
inline int _load_block(const char* source_path, char* dst, size_t size,
                       bool use_o_direct) {
  const int o_direct_flag = use_o_direct ? kODirectFlag : 0;
  const int fd = open(source_path, O_RDONLY | o_direct_flag, 0);
  if (fd < 0) {
    return errno;
  }

  const ssize_t bytes_read = read(fd, dst, size);
  if (bytes_read < 0) {
    // Transient read error: leave the file untouched.
    const int err = errno;
    close(fd);
    return err;
  }
  if (static_cast<size_t>(bytes_read) < size) {
    // Provable short read: the block is genuinely corrupt, so remove it.
    close(fd);
    unlink(source_path);
    return EIO;
  }

  // A close error after a successful full read is harmless: the data is
  // already in the destination buffer, so the load succeeds.
  close(fd);
  return 0;
}

inline void _batch_lookup(const std::vector<const char*>& paths,
                          std::vector<int>& exists_flags) {
  for (size_t i = 0; i < paths.size(); i++) {
    exists_flags[i] = (access(paths[i], F_OK) == 0) ? 1 : 0;
  }
}

// Helper: extract a list[str] of length n into a vector<const char*>.
// Returns false and sets a Python exception on error.
inline bool extract_str_list(PyObject* list, Py_ssize_t n,
                             std::vector<const char*>& out) {
  for (Py_ssize_t i = 0; i < n; i++) {
    out[i] = PyUnicode_AsUTF8AndSize(PyList_GetItem(list, i), nullptr);
    if (out[i] == nullptr) {
      return false;
    }
  }
  return true;
}

// Helper: extract a Py_buffer per element of a list[bytes-like] of length n.
// On success, `out` holds n acquired buffers (caller must PyBuffer_Release
// each). On failure, any buffers already acquired are released before
// returning false, and a Python exception is set.
inline bool extract_buffer_list(PyObject* list, Py_ssize_t n, int flags,
                                std::vector<Py_buffer>& out) {
  for (Py_ssize_t i = 0; i < n; i++) {
    if (PyObject_GetBuffer(PyList_GetItem(list, i), &out[i], flags) != 0) {
      for (Py_ssize_t j = 0; j < i; j++) {
        PyBuffer_Release(&out[j]);
      }
      return false;
    }
  }
  return true;
}

inline void release_buffer_list(std::vector<Py_buffer>& buffers) {
  for (auto& buf : buffers) {
    PyBuffer_Release(&buf);
  }
}

}  // namespace

/// @brief Check file existence for a batch of paths.
/// @param paths list[str] – absolute paths to check.
/// @return list[bool] – True if the corresponding path exists, False otherwise.
/// @note Releases the GIL for the entire batch. File existence via access(2).
static PyObject* batch_lookup(PyObject* /*self*/, PyObject* args) {
  PyObject* path_list;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &path_list)) {
    return nullptr;
  }

  const Py_ssize_t n = PyList_Size(path_list);
  std::vector<const char*> paths(n);
  for (Py_ssize_t i = 0; i < n; i++) {
    paths[i] = PyUnicode_AsUTF8AndSize(PyList_GetItem(path_list, i), nullptr);
    if (paths[i] == nullptr) {
      return nullptr;
    }
  }

  std::vector<int> exists_flags(n);
  {
    Py_BEGIN_ALLOW_THREADS _batch_lookup(paths, exists_flags);
    Py_END_ALLOW_THREADS
  }

  PyObject* result = PyList_New(n);
  if (result == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < n; i++) {
    PyList_SetItem(result, i, PyBool_FromLong(exists_flags[i]));
  }
  return result;
}

/// @brief Store a batch of blocks, each from its own buffer, to disk.
/// @param tmp_paths    list[str] – one temp path per block.
/// @param dest_paths   list[str] – one destination path per block.
/// @param buffers      list[bytes-like] – one source buffer per block.
/// @param use_o_direct bool – whether to open files with O_DIRECT
///                     (default True). Ignored where O_DIRECT is unsupported
///                     by the platform.
/// @note Releases the GIL for the entire batch. Raises on first error.
static PyObject* batch_store_block(PyObject* /*self*/, PyObject* args) {
  PyObject* tmp_paths_obj = nullptr;
  PyObject* dest_paths_obj = nullptr;
  PyObject* buffers_obj = nullptr;
  int use_o_direct = 1;

  if (!PyArg_ParseTuple(args, "O!O!O!|p", &PyList_Type, &tmp_paths_obj,
                        &PyList_Type, &dest_paths_obj, &PyList_Type,
                        &buffers_obj, &use_o_direct)) {
    return nullptr;
  }

  const Py_ssize_t n = PyList_Size(tmp_paths_obj);
  if (PyList_Size(dest_paths_obj) != n || PyList_Size(buffers_obj) != n) {
    PyErr_SetString(
        PyExc_ValueError,
        "tmp_paths, dest_paths and buffers must have the same length");
    return nullptr;
  }

  std::vector<const char*> tmp_paths(n);
  std::vector<const char*> dest_paths(n);

  if (!extract_str_list(tmp_paths_obj, n, tmp_paths)) return nullptr;
  if (!extract_str_list(dest_paths_obj, n, dest_paths)) return nullptr;

  std::vector<Py_buffer> buffers(n);
  if (!extract_buffer_list(buffers_obj, n, PyBUF_SIMPLE, buffers)) {
    return nullptr;
  }

  Py_ssize_t failed_index = -1;
  int failure_errno = 0;

  {
    Py_BEGIN_ALLOW_THREADS for (Py_ssize_t i = 0; i < n; i++) {
      const char* buf = static_cast<const char*>(buffers[i].buf);
      const int err =
          _store_block(tmp_paths[i], dest_paths[i], buf,
                       static_cast<size_t>(buffers[i].len), use_o_direct);
      if (err != 0) {
        failed_index = i;
        failure_errno = err;
        break;
      }
    }
    Py_END_ALLOW_THREADS
  }

  release_buffer_list(buffers);

  if (failed_index >= 0) {
    // PyErr_SetFromErrnoWithFilename() reads the errno to format exception.
    errno = failure_errno;
    return PyErr_SetFromErrnoWithFilename(PyExc_OSError,
                                          dest_paths[failed_index]);
  }

  Py_RETURN_NONE;
}

/// @brief Load a batch of blocks from disk, each into its own buffer.
/// @param source_paths list[str] – one source path per block.
/// @param buffers      list[writable bytes-like] – one destination buffer
///                     per block.
/// @param use_o_direct bool – whether to open files with O_DIRECT
///                     (default True). Ignored where O_DIRECT is unsupported
///                     by the platform.
/// @note Releases the GIL for the entire batch. Raises on first error.
static PyObject* batch_load_block(PyObject* /*self*/, PyObject* args) {
  PyObject* source_paths_obj = nullptr;
  PyObject* buffers_obj = nullptr;
  int use_o_direct = 1;

  if (!PyArg_ParseTuple(args, "O!O!|p", &PyList_Type, &source_paths_obj,
                        &PyList_Type, &buffers_obj, &use_o_direct)) {
    return nullptr;
  }

  const Py_ssize_t n = PyList_Size(source_paths_obj);
  if (PyList_Size(buffers_obj) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "source_paths and buffers must have the same length");
    return nullptr;
  }

  std::vector<const char*> source_paths(n);
  if (!extract_str_list(source_paths_obj, n, source_paths)) return nullptr;

  std::vector<Py_buffer> buffers(n);
  if (!extract_buffer_list(buffers_obj, n, PyBUF_WRITABLE, buffers)) {
    return nullptr;
  }

  Py_ssize_t failed_index = -1;
  int failure_errno = 0;

  {
    Py_BEGIN_ALLOW_THREADS for (Py_ssize_t i = 0; i < n; i++) {
      char* buf = static_cast<char*>(buffers[i].buf);
      const int err =
          _load_block(source_paths[i], buf, static_cast<size_t>(buffers[i].len),
                      use_o_direct);
      if (err != 0) {
        failed_index = i;
        failure_errno = err;
        break;
      }
    }
    Py_END_ALLOW_THREADS
  }

  release_buffer_list(buffers);

  if (failed_index >= 0) {
    // PyErr_SetFromErrnoWithFilename() reads the errno to format exception.
    errno = failure_errno;
    PyErr_SetFromErrnoWithFilename(PyExc_OSError, source_paths[failed_index]);
    // Attach the number of blocks that loaded before the failure so the tier
    // can keep them (partial success). failed_index == count of blocks read OK.
    PyObject *etype, *evalue, *etb;
    PyErr_Fetch(&etype, &evalue, &etb);
    PyErr_NormalizeException(&etype, &evalue, &etb);
    if (evalue != nullptr) {
      PyObject* num = PyLong_FromSsize_t(failed_index);
      if (num != nullptr) {
        PyObject_SetAttrString(evalue, "num_succeeded", num);
        Py_DECREF(num);
      }
    }
    PyErr_Restore(etype, evalue, etb);
    return nullptr;
  }

  Py_RETURN_NONE;
}

/// @brief Compare a batch of loaded blocks with the CRC32 recorded on their
///        files.
/// @param paths   list[str] – one block file per block.
/// @param buffers list[bytes-like] – the loaded bytes of each block.
/// @return list[int] – indices of the blocks whose recorded CRC32 differs
///         from the CRC32 of their buffer. A file without a recorded
///         checksum is not checked and never reported.
/// @note Releases the GIL for the entire batch: one getxattr(2) and one
///       CRC32 per block, with no per-block GIL round trip.
static PyObject* batch_verify_crc32(PyObject* /*self*/, PyObject* args) {
  PyObject* paths_obj = nullptr;
  PyObject* buffers_obj = nullptr;

  if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &paths_obj, &PyList_Type,
                        &buffers_obj)) {
    return nullptr;
  }

  const Py_ssize_t n = PyList_Size(paths_obj);
  if (PyList_Size(buffers_obj) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "paths and buffers must have the same length");
    return nullptr;
  }

  std::vector<const char*> paths(n);
  if (!extract_str_list(paths_obj, n, paths)) return nullptr;

  std::vector<Py_buffer> buffers(n);
  if (!extract_buffer_list(buffers_obj, n, PyBUF_SIMPLE, buffers)) {
    return nullptr;
  }

  std::vector<Py_ssize_t> mismatched;
  {
    Py_BEGIN_ALLOW_THREADS for (Py_ssize_t i = 0; i < n; i++) {
      uint32_t recorded = 0;
      const int status = read_recorded_crc(paths[i], &recorded);
      if (status == 0) {
        continue;
      }
      const auto* buf = static_cast<const unsigned char*>(buffers[i].buf);
      if (status < 0 ||
          recorded != crc32_of(buf, static_cast<size_t>(buffers[i].len))) {
        mismatched.push_back(i);
      }
    }
    Py_END_ALLOW_THREADS
  }

  release_buffer_list(buffers);

  PyObject* result = PyList_New(static_cast<Py_ssize_t>(mismatched.size()));
  if (result == nullptr) {
    return nullptr;
  }
  for (size_t i = 0; i < mismatched.size(); i++) {
    PyObject* idx = PyLong_FromSsize_t(mismatched[i]);
    if (idx == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    PyList_SetItem(result, static_cast<Py_ssize_t>(i), idx);
  }
  return result;
}

static PyMethodDef fs_io_C_methods[] = {
    {"batch_lookup", batch_lookup, METH_VARARGS,
     "batch_lookup(paths: list[str]) -> list[bool]\n"
     "\n"
     "Check file existence for a batch of paths."},
    {"batch_store_block", batch_store_block, METH_VARARGS,
     "batch_store_block(tmp_paths: list[str], dest_paths: list[str],\n"
     "                  buffers: list[bytes-like],\n"
     "                  use_o_direct: bool = True) -> None\n"
     "\n"
     "Store a batch of blocks, each from its own buffer, to disk. Raises on "
     "first error."},
    {"batch_load_block", batch_load_block, METH_VARARGS,
     "batch_load_block(source_paths: list[str],\n"
     "                 buffers: list[writable bytes-like],\n"
     "                 use_o_direct: bool = True) -> None\n"
     "\n"
     "Load a batch of blocks from disk into corresponding buffers. "
     "Raises on first error."},
    {"batch_verify_crc32", batch_verify_crc32, METH_VARARGS,
     "batch_verify_crc32(paths: list[str], buffers: list[bytes-like])\n"
     "    -> list[int]\n"
     "\n"
     "Indices of the blocks whose recorded CRC32 xattr differs from the "
     "CRC32 of their buffer. Files without a recorded checksum are not "
     "checked."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef fs_io_C_module = {
    PyModuleDef_HEAD_INIT, "fs_io_C", "Filesystem helpers for KV offload", -1,
    fs_io_C_methods,
};

PyMODINIT_FUNC PyInit_fs_io_C(void) { return PyModule_Create(&fs_io_C_module); }

}  // extern "C"
