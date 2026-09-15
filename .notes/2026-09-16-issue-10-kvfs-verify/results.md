# 이슈 #10 구현·검증 기록

## 구현 단계

- 기준: `5700ec2b8b`, 승인 계획 `plan.md`, 최신 레시피 `icmt-83bf7d7c`.
- 제품 변경: `deploy/gb10-cluster/dsv41/{kvfs_verify.py,test_kvfs_verify.py,README.md}`.
- 부가 변경: 이 메모와 계획, `.gitignore`의 메모 디렉터리 예외.
- 표준 라이브러리만 사용한다. 프로젝트에서 `re`를 금지하므로 경로는 문자열 분리·문자 검증으로 해석한다.
- 안전 조치: 기본 보고 모드, 서버 상태 불확실 시 삭제 거부, 링크를 따르지 않는 fd 접근, 삭제 직전 분류·파일 identity 재검사, 명시 후보 목록 출력. no_xattr/unknown/tmp는 삭제하지 않는다.
- 로컬 환경: uv로 생성한 Python 3.12 `.venv`, lint 의존성·pytest 설치, pre-commit/commit-msg 훅 설치.

### 실행 결과

```text
.venv/bin/python -m pytest -c /dev/null -p no:cacheprovider deploy/gb10-cluster/dsv41/test_kvfs_verify.py -q
23 passed, 4 skipped in 0.08s
```

macOS의 `os.setxattr` 부재로 실제 xattr 기반 분류/삭제 매개변수 케이스 4개는 skip했다. CRC 재읽기·지속 손상·fadvise는 속성 읽기 주입으로 로컬에서도 검사했다. 실제 Linux xattr 테스트 통과는 아직 주장하지 않는다.

```text
.venv/bin/pre-commit run --files deploy/gb10-cluster/dsv41/kvfs_verify.py deploy/gb10-cluster/dsv41/test_kvfs_verify.py deploy/gb10-cluster/dsv41/README.md .gitignore .notes/2026-09-16-issue-10-kvfs-verify/plan.md
All applicable hooks passed.

git diff --check
Passed.

.venv/bin/python deploy/gb10-cluster/dsv41/kvfs_verify.py --help
Exit 0; root/delete/limit/sample/progress/fadvise options displayed.
```

## 구현 종료 시 인계 사항 (이하 검증에서 완료)

노드에는 아직 접근·반영하지 않았다. head pytest, 라이브/옛 저장소 전수 보고, 손상 사본 검출·삭제 거부, 실행 전후 메모리 및 운영 상태는 다음 단계에서 수행하고 여기에 추가한다. 실제 kvfs 파일 삭제는 현재 0이며 이후 검증에서도 0을 유지한다. 모델·엔진 변경은 없다.

## validate 결과 — 통과

2026-09-16 02:25~02:41 KST, head `gx10-6040`에서 실행했다. 기준 ref
`5700ec2b8b167a2f699c265ada89390e3ed85334` 대비 변경을 최초 요청·승인 계획과
대조했다. 엔진/C/GC/서버 설정 변경은 없고 실제 kvfs 삭제는 **0개**다.

| 대상 | 검사 파일 | 정상 | 크기/CRC 불일치 | 삭제 | 시간 | 처리량 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 라이브 `/mnt/kvdisk/kv/dsv41` | 1,045,649 | 1,045,649 | 0 / 0 | 0 | 590.283 s | 169.91 MiB/s |
| 옛 `~/dsv41-prep/kvfs` | 379,057 | 379,057 | 0 / 0 | 0 | 134.614 s | 268.76 MiB/s |
| 손상 사본, 보고 모드 | 3 | 2 | 0 / 1 | 0 | 0.020 s | 23.14 MiB/s |
| 손상 사본, `--delete` | 3 | 2 | 0 / 1 | 0 | 0.006 s | 80.34 MiB/s |

두 실저장소의 no_xattr/unknown/vanished/skipped_tmp도 모두 0이었다.
검사는 원자적 스냅샷이 아닌 순회 전수 검사다. 라이브 시작 전 실행·대기 요청은
metrics에서 모두 0이었고 status의 health는 200이었다. 현재 ctl status는 요청 수를
출력하지 않으므로 `vllm:num_requests_running/waiting`으로 유휴를 직접 확인했다.

### 독립 리뷰 및 작은 수정

- 리뷰 서브태스크: `tsk_7631aad2-1799-4f48-8457-133d8f041b23`,
  워크트리 `i10-review-a6631bfc`. 삭제 경계·CRC 경쟁·형식 해석을 독립 검토했다.
- 지적 1건: 일반 read가 relatime 마운트에서 atime을 갱신하면 GC의 용량 정리
  순서에 영향을 준다. 실측 mount 옵션은 라이브 `noatime`, 옛 루트 `relatime`이었다.
- `a76e5a6509`에서 Linux `O_NOATIME`을 추가하고 소유권상 EPERM일 때 경고 후
  일반 읽기로 전환한다. 실제 atime 보존 테스트와 EPERM fallback 테스트를 추가했다.
  수정 hunk와 직접 영향 테스트를 재검토했다. 그 외 실질 지적은 없었다.
- 이미 실행 중이던 라이브 전수 검사는 `cdbc8e4fbc`의 코드로 완료했다.
  라이브 마운트 자체가 noatime이므로 지적된 부작용은 없다. 수정본은 head에서
  29개 테스트와 옛 저장소 전수 검사·사본 검사로 검증했다. 이 한 플래그 변경
  때문에 라이브 전수 검사를 반복하지 않았다.

### 테스트·린트 및 노드 반영

- 초기 head pytest: **27 passed in 0.12s**, skip 없음.
- 수정 후 로컬 pytest: **24 passed, 5 skipped**. macOS xattr/O_NOATIME 부재에
  해당하는 skip이며 head에서는 모두 실행했다.
- 최종 head pytest: **29 passed in 0.13s**, skip 없음.
- `.venv/bin/pre-commit run --from-ref 5700ec2b8b167a2f699c265ada89390e3ed85334 --to-ref HEAD`:
  적용 대상 훅 전부 통과. `git diff --check` 통과. torch/vllm import 부재는 소스
  검사와 테스트 모듈의 sys.modules 단언으로 확인했다. 모델 변경이 없어 모델 eval은
  하지 않았으며 서버 상태를 읽기만 했다.
- head에는 `git format-patch` / `git am`으로 스크립트·테스트만 반영했다.
  README는 `git apply --check`에서 기존 내용과 불일치하여 레시피 허용대로
  노드 패치에서 제외했다. 실패한 am/rebase 상태를 남기지 않았다.
  노드 커밋은 `12a57dc573` → `88744c769e`, 최종 체크아웃은 깨끗하다.
- `~/dsv41-prep/kvfs_verify.py`는 기존 파일이 없음을 확인한 뒤
  `~/vllm-dsv41/deploy/gb10-cluster/dsv41/kvfs_verify.py`로 심링크를 만들었다.
  스크립트·테스트의 로컬/head SHA-256은 각각 일치한다(아래 출력).
- head pytest는 fresh ext4 경로 `i10-pytest.0d3aKA`, `i10-pytest.PDst6K`를
  각각 한 번만 basetemp로 사용했다. 실행 전 realpath·빈 디렉터리를 확인했다.
  pytest의 삭제 성공 케이스는 이곳의 생성 파일만 대상으로 했다.

### 메모리·운영 상태

| 시점 | MemAvailable (kB) | Cached (kB) |
| --- | ---: | ---: |
| 라이브 시작 전 | 8,634,724 | 4,886,468 |
| 라이브 완료 후 | 9,509,824 | 3,343,160 |
| 옛 저장소 시작 전 | 9,492,372 | 3,355,096 |
| 옛 저장소 완료 후 | 9,582,180 | 3,323,920 |
| 최종 head 확인 | 9,610,916 | 3,332,868 |

읽은 약 133 GiB가 페이지 캐시로 잔류하지 않았다. 이 전후 수치만으로 모든 메모리
변화를 fadvise의 효과로 단정하지는 않는다. 약 30~60초 간격의 관측에서 health는
계속 200이었다. 최종 4대 serve/cap active·1989 MHz, watchdog active/ATTENTION
없음, override 4대 모두 0바이트. 서버·타이머·캡 재시작, sysctl/swap/drop_caches
조작은 없었다.

### 사본·삭제 감사

- `tempfile.mkdtemp`로 `~/dsv41-prep/i10-copy.l0mq8pb3`을 새로 만들고,
  realpath가 준비 디렉터리 바로 아래임을 확인했다. 실파일 3개와 config를
  `cp --preserve=xattr`로 상대 경로 유지 복사하고 원본/사본 xattr 일치도 확인했다.
- 사본의 첫 파일만 첫 바이트를 XOR 1로 변경했다. 보고 종료 1, `crc_mismatch=1`.
  활성 서버 아래 `--delete` 종료 3, `deleted=0`, 세 파일이 모두 남아 있음을
  cleanup 전 목록으로 확인했다. CLI parser/help에 서버 검사 우회 옵션은 없다.
- 02:40:44 KST에 위 사본의 realpath·전체 목록(일반 파일 4개·디렉터리 6개)과
  심링크 없음 확인 후 **명시 절대 경로 하나**에 `rm -r --`를 실행했다.
  대상 부재·종료 0을 확인했다. 실저장소·옛 저장소와 공유 준비 디렉터리는 삭제하지
  않았다. 안전 확인창은 발생하지 않았다. pytest fixture 디렉터리는 그대로 남겼다.

## 실기 출력 원문

### 시작 status

```text
gx10-6040 r0: active used 113.39 GiB avail 8.24 GiB cap active 1989MHz
gx10-f323 r1: bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
active used 113.93 GiB avail 7.70 GiB cap active 1989MHz
gx10-37cc r2: bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
active used 114.11 GiB avail 7.52 GiB cap active 1989MHz
gx10-27c4 r3: bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
active used 113.79 GiB avail 7.84 GiB cap active 1989MHz
health: 200
watchdog: active, attention: none
```

### 라이브 시작 전 유휴·메모리

```text
2026-09-16T02:26:37+09:00
vllm:num_requests_running{engine="0",model_name="deepseek-v4.1-flash"} 0.0
vllm:num_requests_waiting{engine="0",model_name="deepseek-v4.1-flash"} 0.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="deepseek-v4.1-flash",reason="capacity"} 0.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="deepseek-v4.1-flash",reason="deferred"} 0.0
MemAvailable:    8634724 kB
Cached:          4886468 kB
active
```

### head 초기 pytest

```text
...........................                                              [100%]
27 passed in 0.12s
```

### 라이브 전수 검사

```text
warning: server running; report mode only
kvfs_verify: inspected=50000
kvfs_verify: inspected=100000
kvfs_verify: inspected=150000
kvfs_verify: inspected=200000
kvfs_verify: inspected=250000
kvfs_verify: inspected=300000
kvfs_verify: inspected=350000
kvfs_verify: inspected=400000
kvfs_verify: inspected=450000
kvfs_verify: inspected=500000
kvfs_verify: inspected=550000
kvfs_verify: inspected=600000
kvfs_verify: inspected=650000
kvfs_verify: inspected=700000
kvfs_verify: inspected=750000
kvfs_verify: inspected=800000
kvfs_verify: inspected=850000
kvfs_verify: inspected=900000
kvfs_verify: inspected=950000
kvfs_verify: inspected=1000000
ok: count=1045649 bytes=105164853248
  /mnt/kvdisk/kv/dsv41/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g15/000011abb829a6535661cc025b42187adeccaacacfb23bf02ce05ba9714809c1.bin
  /mnt/kvdisk/kv/dsv41/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g15/00001210ee607cad8a1b5d51b0215e9dd3ba5ef26aea72b54c5d9c6793d3860c.bin
  /mnt/kvdisk/kv/dsv41/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g17/000011abb829a6535661cc025b42187adeccaacacfb23bf02ce05ba9714809c1.bin
  /mnt/kvdisk/kv/dsv41/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g17/00001210ee607cad8a1b5d51b0215e9dd3ba5ef26aea72b54c5d9c6793d3860c.bin
  /mnt/kvdisk/kv/dsv41/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/04_g17/000040c6361616d8d5f03f77a2a6564bd73dcd9024a506712e20d0678a20e243.bin
size_mismatch: count=0 bytes=0
no_xattr: count=0 bytes=0
crc_mismatch: count=0 bytes=0
unknown: count=0 bytes=0
vanished: count=0 bytes=0
skipped_tmp: count=0 bytes=0
no_xattr/unknown are unverified; live scans are not snapshots.
kvfs_verify: root=/mnt/kvdisk/kv/dsv41 files=1045649 ok=1045649 size_mismatch=0 no_xattr=0 crc_mismatch=0 unknown=0 vanished=0 skipped_tmp=0 deleted=0 secs=590.283 mib_s=169.91
exit_code=0
2026-09-16T02:36:38+09:00
MemAvailable:    9509824 kB
Cached:          3343160 kB
```

### head 최종 pytest

```text
.............................                                            [100%]
29 passed in 0.13s
```

### 옛 저장소 전수 검사

```text
2026-09-16T02:37:26+09:00
MemAvailable:    9492372 kB
Cached:          3355096 kB
warning: server running; report mode only
kvfs_verify: inspected=50000
kvfs_verify: inspected=100000
kvfs_verify: inspected=150000
kvfs_verify: inspected=200000
kvfs_verify: inspected=250000
kvfs_verify: inspected=300000
kvfs_verify: inspected=350000
ok: count=379057 bytes=37936291840
  /home/nacyot/dsv41-prep/kvfs/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/06_g15/000064fadd9c24a82f8561745160fc8fff8e455a20f9aee205b8522abd277b31.bin
  /home/nacyot/dsv41-prep/kvfs/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/06_g17/000064fadd9c24a82f8561745160fc8fff8e455a20f9aee205b8522abd277b31.bin
  /home/nacyot/dsv41-prep/kvfs/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/0d_g15/0000d37c08051be7152520f26977c98ed3ef534da776745b19669f8652b51477.bin
  /home/nacyot/dsv41-prep/kvfs/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/0d_g17/0000d37c08051be7152520f26977c98ed3ef534da776745b19669f8652b51477.bin
  /home/nacyot/dsv41-prep/kvfs/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/17_g15/0001754b9c75c90fc6e485366e43557ed697f916fb400913937d3d218ac9af38.bin
size_mismatch: count=0 bytes=0
no_xattr: count=0 bytes=0
crc_mismatch: count=0 bytes=0
unknown: count=0 bytes=0
vanished: count=0 bytes=0
skipped_tmp: count=0 bytes=0
no_xattr/unknown are unverified; live scans are not snapshots.
kvfs_verify: root=/home/nacyot/dsv41-prep/kvfs files=379057 ok=379057 size_mismatch=0 no_xattr=0 crc_mismatch=0 unknown=0 vanished=0 skipped_tmp=0 deleted=0 secs=134.614 mib_s=268.76
exit_code=0
2026-09-16T02:39:41+09:00
MemAvailable:    9582180 kB
Cached:          3323920 kB
```

### 손상 사본 준비

```text
copy_root=/home/nacyot/dsv41-prep/i10-copy.l0mq8pb3
copy_file=/home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g15/000011abb829a6535661cc025b42187adeccaacacfb23bf02ce05ba9714809c1.bin
copy_file=/home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g15/00001210ee607cad8a1b5d51b0215e9dd3ba5ef26aea72b54c5d9c6793d3860c.bin
copy_file=/home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g17/000011abb829a6535661cc025b42187adeccaacacfb23bf02ce05ba9714809c1.bin
corrupted=/home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g15/000011abb829a6535661cc025b42187adeccaacacfb23bf02ce05ba9714809c1.bin
```

### 손상 사본 보고

```text
warning: server running; report mode only
ok: count=2 bytes=217088
  /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g15/00001210ee607cad8a1b5d51b0215e9dd3ba5ef26aea72b54c5d9c6793d3860c.bin
  /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g17/000011abb829a6535661cc025b42187adeccaacacfb23bf02ce05ba9714809c1.bin
size_mismatch: count=0 bytes=0
no_xattr: count=0 bytes=0
crc_mismatch: count=1 bytes=139264
  /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g15/000011abb829a6535661cc025b42187adeccaacacfb23bf02ce05ba9714809c1.bin
unknown: count=0 bytes=0
vanished: count=0 bytes=0
skipped_tmp: count=0 bytes=0
no_xattr/unknown are unverified; live scans are not snapshots.
kvfs_verify: root=/home/nacyot/dsv41-prep/i10-copy.l0mq8pb3 files=3 ok=2 size_mismatch=0 no_xattr=0 crc_mismatch=1 unknown=0 vanished=0 skipped_tmp=0 deleted=0 secs=0.020 mib_s=23.14
exit_code=1
```

### 손상 사본 삭제 거부

```text
warning: server running; report mode only
ok: count=2 bytes=217088
  /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g15/00001210ee607cad8a1b5d51b0215e9dd3ba5ef26aea72b54c5d9c6793d3860c.bin
  /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g17/000011abb829a6535661cc025b42187adeccaacacfb23bf02ce05ba9714809c1.bin
size_mismatch: count=0 bytes=0
no_xattr: count=0 bytes=0
crc_mismatch: count=1 bytes=139264
  /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g15/000011abb829a6535661cc025b42187adeccaacacfb23bf02ce05ba9714809c1.bin
unknown: count=0 bytes=0
vanished: count=0 bytes=0
skipped_tmp: count=0 bytes=0
no_xattr/unknown are unverified; live scans are not snapshots.
kvfs_verify: root=/home/nacyot/dsv41-prep/i10-copy.l0mq8pb3 files=3 ok=2 size_mismatch=0 no_xattr=0 crc_mismatch=1 unknown=0 vanished=0 skipped_tmp=0 deleted=0 secs=0.006 mib_s=80.34
exit_code=3
```

### 사본 정리 전 목록 및 CLI

```text
/home/nacyot/dsv41-prep/i10-copy.l0mq8pb3
d /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3
d /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0
d /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000
d /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g15
f /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g15/00001210ee607cad8a1b5d51b0215e9dd3ba5ef26aea72b54c5d9c6793d3860c.bin
f /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g15/000011abb829a6535661cc025b42187adeccaacacfb23bf02ce05ba9714809c1.bin
d /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g17
f /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b_r0/000/01_g17/000011abb829a6535661cc025b42187adeccaacacfb23bf02ce05ba9714809c1.bin
d /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b
f /home/nacyot/dsv41-prep/i10-copy.l0mq8pb3/_home_nacyot_models_DeepSeek-V4.1-Flash_06d2a1a0be2b/config.json
usage: kvfs_verify.py [-h] [--delete] [--limit LIMIT] [--sample SAMPLE]
                      [--progress] [--fadvise | --no-fadvise]
                      [root]

Inspect filesystem KV blocks without importing vLLM or modifying the store.

positional arguments:
  root

options:
  -h, --help            show this help message and exit
  --delete
  --limit LIMIT
  --sample SAMPLE       inspect every Kth bin in sorted traversal (default: 1)
  --progress
  --fadvise, --no-fadvise
```

### 사본 정리 후 시각 (종료 0)

```text
2026-09-16T02:40:44+09:00
```

### 최종 status

```text
gx10-6040 r0: active used 112.46 GiB avail 9.16 GiB cap active 1989MHz
gx10-f323 r1: bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
active used 113.92 GiB avail 7.71 GiB cap active 1989MHz
gx10-37cc r2: bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
active used 114.16 GiB avail 7.47 GiB cap active 1989MHz
gx10-27c4 r3: bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
active used 113.79 GiB avail 7.84 GiB cap active 1989MHz
health: 200
watchdog: active, attention: none
```

### 최종 head 및 소스 해시

```text
2026-09-16T02:40:58+09:00
0 /home/nacyot/dsv41-prep/dsv41-override.env
active
attention=none
MemAvailable:    9610916 kB
Cached:          3332868 kB
e0a0c4a216586acbdccf51e31e96726091ebe118c11cb63db77fcda1dc239ffe  deploy/gb10-cluster/dsv41/kvfs_verify.py
fd68082d41ed55fa4e7f51d2e2161093d616b936e2e37f0e9220eabdd3e4116a  deploy/gb10-cluster/dsv41/test_kvfs_verify.py
/home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/kvfs_verify.py
```

### 워커 override 크기

```text
bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
gx10-f323
0 /home/nacyot/dsv41-prep/dsv41-override.env
bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
gx10-37cc
0 /home/nacyot/dsv41-prep/dsv41-override.env
bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
gx10-27c4
0 /home/nacyot/dsv41-prep/dsv41-override.env
```
