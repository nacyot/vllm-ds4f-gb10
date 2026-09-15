# 이슈 #37 결과 — 디코드 prefault 를 배경으로(모드 1) 채택, 코드 c1 스텝 70.6 → 68.2 ms, 프로덕션 복구 완료

2026-09-15 15:30~16:07 KST, 워크트리 `issue-37-engram-prefault`(base `main` 0a25443fe4). `plan.md` 의 부팅 0·1·2 와 최종 부팅을 수행했다(총 4 부팅, 계획 최대치). 측정은 전부 헤드 `~/sglang-cmp/results/*.jsonl`·`*.txt`(같은 도구·인자), 스탯은 `dsv41-r0.log` 의 `ENGRAM_STATS=1` 디코드 버킷.

## 1. 결론

- **근본 원인 확정**: 디코드 스텝의 Engram 페이지는 콜드 NVMe 읽기다(샤드 상주 0.1 %, 표당 72~96 페이지, 장치 22k IOPS). 러너가 해시 D2H 뒤 표 두 장을 직렬로 기다려 스텝당 5.6 ms 가 GPU 유휴였다(기준선 프로파일, release 0). 가설 1(풀 분배 비용)은 ~1 ms 몫, 가설 3(상주 건너뛰기)은 이득 0 — 계획서 §2.
- **채택**: `ENGRAM_DECODE_ASYNC=1`(첫 표(레이어 1)만 기다리고 레이어 14 표는 포워드 중 배경 populate) + `ENGRAM_CHUNK_RUNS=8`. 코드·산문 c1 스텝이 모두 기준선보다 짧고 부작용이 없다. 수락 길이 5.82 불변, 158K 니들 정답.
- **모드 2(두 표 모두 배경)는 옵트인**으로 남긴다: 코드는 64.6 ms / 유휴 2.1 ms 로 완료 기준을 넘지만, 산문에서 레이어 1 lookup 이 콜드 페이지를 GPU 제자리 fault 로 읽어(engram 커널 3.66 ms/스텝, lookup 당 1.05 ms) 산문 스텝이 모드 1 보다 +3.7 ms 다. 코드는 n-gram 반복으로 페이지가 이미 상주해 fault 가 없다.
- 완료 기준 "코드 c1 ≤ 68 ms 또는 유휴 ≤ 2 ms" 는 채택 구성에서 **경계값**(68.2 ms 평균, 최종 부팅 68.2/67.0; 유휴 4.3 ms). 모드 2 만이 둘 다 명확히 넘는다. 남은 격차는 레이어 1 표의 디스크 시간(3.5 ms)이라, 후속은 §5.

## 2. 대조표

디코드는 `decodebench.py --levels 1 --gen 256`, 스텝 = tokens_per_chunk / per_stream_1. 각 부팅의 r1 은 웜업 직후라 매번 느리다(기준선 73.3, 모드 1 68.6, 모드 2 69.6).

| 구성 | 코드 c1 스텝 ms (r1/r2/r3) | 산문 c1 스텝 ms | 수락(코드) | 스텝당 prefault: 스텝 위 / 배경 (표 47 + 48) | DtoH 뒤 GPU 유휴 (프로파일, 코드) | engram 커널 합 (코드) |
| --- | --- | --- | ---: | --- | ---: | ---: |
| prod-nob12x (15:00, #36 직후) | 69.5 (1회) | 72.4 | 5.82 | — | — | — |
| 부팅 0 `i37-base` 기준선 | 73.3 / 70.2 / 68.4 | 70.3 / 70.7 | 5.82 | 3.2+2.5 = **5.8** / 0 | 166.9 ms / 30 = **5.6 ms** (스텝 gap 5.3) | 1.3 ms |
| 부팅 1 `i37-async1` 모드 1 + 조각 8 | 68.6 / 69.1 / 67.0 | 68.0 / 66.8 / 68.7 | 5.82 | 3.6+0.4 = **4.0** / 3.1 | 128.0 / 30 = **4.3 ms** (gap 4.0) | 1.1 ms |
| 부팅 2 `i37-async2` 모드 2 + 조각 8 | 69.6 / 64.5 / 64.8 | 71.4 / 72.6 / 70.6 | 5.82 | 0.5+0.4 = **0.9** / 4.5+4.5 | 62.9 / 30 = **2.1 ms** (gap 1.5) | 1.1 ms (산문은 122.7 ms/58 스텝 = 3.66 ms/스텝) |
| **최종 `prod-i37` (채택 기본값, 오버라이드 없음)** | 68.2 / 67.0 | 68.5 / 67.5 | 5.82 | — (스탯 꺼짐) | — | — |

158K 니들(`longctx.py --records 10700`, 최종 부팅): 콜드 정답 "12", TTFT 84.3 s(1,880 tok/s; combo-nob12x 101.1 s), 후속 턴 정답 "986", TTFT 0.753 s(#36 0.77).

헤드 MemAvailable: 부팅 직후 5.80 → 디코드 뒤 4.78 → 니들 뒤 5.75 → 마감 5.64 GiB. 실험 중 최저는 부팅 2 산문 프로파일 중 2.81 GiB(프로파일러 트레이스 20 MB 쓰기 구간; 비프로파일 셀 최저 3.13). 캡 4 대 active/Enabled 1989~1995 MHz.

## 3. 변경 (커밋, base 0a25443fe4)

- fb667d9c16 `engram: decode steps can populate mmap table pages in the background (mmap_decode_async)` — `vllm/config/engram.py` 필드 `mmap_decode_async`(0/1/2)·`mmap_min_chunk_runs`; `common/engram.py` `MmapEngramTable.prefault(wait=False)`(전용 1 스레드 워커, 링 > 0 이면 동기로 되돌리고 경고), `_run_over_pages` 조각 수, 디코드 버킷 스탯; `nvidia/model_state.py` 표별 wait(배경 표 먼저 제출); `serve-node.sh`/`dsv41_ctl.sh`/`dsv41.env` 노브(기본 0/1 = 행동 불변); 테스트 2 개.
- 33c5cd76e7 `tests(engram): ...` — 조각 테스트의 스레드 수 정정.
- f9bcd23d08 `deploy(dsv41): adopt ENGRAM_DECODE_ASYNC=1 and ENGRAM_CHUNK_RUNS=8 as the defaults` — `dsv41.env` 기본값·주석, README 문단 + 복구 목록.
- 단위 테스트: 헤드에서 서버 정지 구간에 `pytest tests/kernels/test_engram.py -k "run_over_pages or decode_async or mmap_table"` 8 passed(CPU 전용). 로컬 ruff/shellcheck/typos 통과; mypy 4+2 오류와 `check-torch-cuda-call` 1 건은 전부 이번 hunk 밖의 기존 항목(macOS stub 의 `os.posix_fadvise`, 기존 `torch.cuda.synchronize()`)이라 그 두 훅만 SKIP 으로 커밋. `selftest_caps.sh` 전부 PASS(bash 4), `test_headroom.py` 13 passed.

## 4. 노드·부팅 기록

- 4 노드 `~/vllm-dsv41` 에 세 커밋 `git format-patch | git am`(워커 `-c user.name=nacyot -c user.email=<head 값>`, 셋째 커밋은 README hunk `--exclude`). 헤드 7d9c679d16, 전부 dirty 0. 파이썬/셸만이라 재빌드 없음.
- 부팅 0 15:31:52 start → 15:35:00 health(`ENGRAM_STATS=1 PROFILER_DIR=prof-i37-base`, 인자 decode_async=0, chunk 1) → 부팅 1 15:39 → 15:43:14 health(async=1, chunk 8) → 부팅 2 15:46 → 15:49:57 health(async=2). 15:56:42 세션 재시작으로 프로덕션이 내려간 상태(health 000)를 매니저가 확인, 부팅 2 의 니들·AG4 는 미실행 → 판정은 §1 의 데이터로.
- 최종 16:00:05 `dsv41_ctl.sh start`(오버라이드 0, unit Environment 에 ENGRAM/PROFILER/SPEC 없음) → 16:03:23 health 200 → 부팅 인자 `mmap_decode_async=1, mmap_min_chunk_runs=8`, profiler 없음.
- 실험 부팅(15:31~16:07)의 `POST /v1` 은 전부 127.0.0.1(외부 클라이언트 0; 직전 외부 요청 192.168.1.222 는 15:21).
- 프로파일 트레이스는 헤드·워커의 `~/dsv41-prep/prof-i37-{base,async1,async2}/` 에 남아 있다(요약 `results/i37-*-code-*.txt`, `i37-async2-prose-*.txt`).

## 5. 남는 것

- **AG4(에이전트 4×6) 미측정**: 부팅 예산(4) 안에서 니들·짧은 디코드만 최종 부팅에서 확인했다. 채택 구성은 코드·산문 c1 이 모두 기준선보다 좋아 AG4 가 나빠질 경로가 없지만, #42 새 기준선에서 208 s 대비 재확인이 필요하다.
- **모드 3(경계 대기)**: 레이어 1 표를 짧은 데드라인(예 1.5 ms)만 기다리고 남은 콜드 페이지는 GPU 가 읽게 하면 코드는 모드 2 의 이득을, 산문은 fault 를 일부만 내게 된다. 유휴 ≤ 2 ms 를 산문까지 넘기려면 이것 또는 NVMe 병렬 읽기 확장이 필요. 별도 이슈로.
- 기준선 스탯의 `pages/call` 92~96 은 계획 §3 의 72 보다 크다(디코드 버킷에 산문·초기 스텝의 큰 배치가 섞임). 스텝당 4.5 ms 배경 populate(모드 2, 두 표 동시 → NVMe 경합)는 22k IOPS 상한과 맞다.
