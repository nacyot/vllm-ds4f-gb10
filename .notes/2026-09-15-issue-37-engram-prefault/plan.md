# 이슈 #37 계획 — 디코드 스텝의 Engram prefault 는 콜드 NVMe 읽기다: 임계 경로에서 겹치기(비동기)로 걷어낸다

작성 2026-09-15 15:40 KST, 워크트리 `issue-37-engram-prefault`(base `main` 0a25443fe4). 매니저 코멘트 icmt-06d96ae9 의 순서(재측정 → 게이트 A/B → 판정)를 따르되, 게이트의 내용은 이 계획 단계의 실측으로 바꾼다. 코드는 아직 건드리지 않았다. 조사는 전부 읽기 전용(로컬 파일, 헤드 `~/sglang-cmp/results/`, `dsv41-r0.log`, `systemctl show`) + 헤드에서 돌린 소형 프로브 두 개(별도 프로세스, 페이지 캐시 ~10 MB 읽기, 라이브 서버 무간섭: `prefault_probe.py.txt`, `nvme_probe.py.txt` 이 디렉터리에 동봉(일회성 프로브라 .txt)).

## 1. 증상

- 코드 동시 1 스텝(= 5.82 / per_stream_1): 현재 프로덕션 기본값(prod-nob12x, 15:00) 69.5 ms, T2W 65.6~68.0 ms. 남은 격차 2~4 ms(매니저 §1).
- 프로파일의 GPU 유휴는 `Memcpy DtoH (Device -> Pageable)` 바로 뒤에 몰려 있다: prof-base(release 3, c1) 30회 309.9 ms = 10.3 ms/스텝; **prof-c2(오늘 11:09, release 0 인 full-combo 부팅, c2) 31회 378.5 ms = 12.2 ms/스텝**. 즉 release 0 으로도 DtoH 뒤 유휴는 그대로다(가설 2 는 3.2 ms 몫만 가져갔다).
- 그 DtoH 는 `vllm/models/deepseek_v4_1/nvidia/model_state.py:136-145` `_prefault_engram_rows` 의 `.cpu()` 이고, 그 뒤 `:146-147` 이 표 두 장(레이어 1, 14)의 `embed_tokens.prefault` 를 **차례로, 동기로** 부른다. `.cpu()` 는 직전 스텝의 GPU 작업이 끝나야 돌아오므로, 그 뒤 포워드 런치까지 CPU 가 쓰는 시간이 곧 GPU 유휴다.

## 2. 근본 원인 (실측)

### 2-1. 디코드 스텝의 페이지는 사실상 전부 콜드다

헤드 `mincore` 로 두 샤드(각 94.6 GiB, 표는 그 안 23.6 GiB/랭크)의 페이지 캐시 상주를 셌다(`prefault_probe.py.txt`, 15:30, 프로덕션 가동 중):

| 샤드 | 상주 | 비율 |
| --- | ---: | ---: |
| model-00047 (레이어 1) | 0.07 GiB / 94.6 | 0.1 % |
| model-00048 (레이어 14) | 0.03 GiB / 94.6 | 0.0 % |

랭크당 표 두 장 47 GiB 를 헤드 여유 5~7 GiB 가 담을 수 없고, 해시는 96M 행 위에 무작위라 재사용도 없다. `ENGRAM_RELEASE=0` 은 "회수를 커널에 맡김"이지 상주를 만들지 못한다. **따라서 매니저 가설 3(상주 페이지 건너뛰기 비트맵)은 이득이 0 이다** — 건너뛸 상주 페이지가 없다.

### 2-2. 표당 prefault 비용은 NVMe 무작위 4 KiB 읽기 그 자체다

같은 프로브로 디코드 크기(스텝 6 토큰 × 랭크당 6 열 = 표당 36 행 → weight 36 + scale 36 = **72 페이지, 72 run**)를 실제 `madvise(MADV_POPULATE_READ)` 로 재봤다(8 회 중앙값, 두 샤드 동일 경향):

| 경로 | 콜드(첫 방문) | 같은 페이지 다시(상주) |
| --- | ---: | ---: |
| 현재 코드 = 풀, 72 run > 64 → `array_split(72)` 조각 72 개 | **5.9 ms** | 1.3 ms |
| 인라인(조각 없음, 순차) | 17.2 ms | 0.15 ms |
| 풀, 조각 8 개 | 4.9 ms | 0.7 ms |
| 144 페이지(c2 급) 풀 72→128 조각 | 9.6 ms | 1.8 ms |

NVMe 원시 성능(`nvme_probe.py.txt`, O_DIRECT pread 4 KiB 무작위, 장치 ESL01TBTLCZ 루트 디스크 92 % 사용): 1 스레드 111 µs/읽기 9.0k IOPS, 8 스레드 이상 **22k IOPS 상한** → 72 읽기의 하한 3.2 ms, 144 읽기 6.5 ms.

정리하면 스텝당(표 2 장 직렬) ≈ 2 × 5.9 ≈ **12 ms** 가 DtoH 뒤 유휴로 들어간다. prof-c2 의 12.2 ms/스텝과 맞는다. 구성은:

- 디스크 읽기 ≥ 3.2 ms/표(장치 상한, 코드로 못 줄임)
- 풀 분배 파이썬 비용 ≈ 1.2 ms/표(상주 케이스 풀 1.3 vs 인라인 0.15; 가설 1 은 맞지만 몫이 작다)
- 조각 72 → 8 로 줄이면 콜드 −0.9 ms/표

**가설 1 의 처방(인라인 임계값 64 → 512)은 반대로 표당 +11 ms 다**(콜드 순차 17 ms). 채택하지 않는다.

### 2-3. 결론: 기다리는 구조가 문제다

읽기 자체를 없앨 수 없으므로(T2W 는 표를 pinned RAM 에 두어 애초에 읽기가 없다, 매니저 §2), 남은 길은 **읽기를 포워드와 겹치는 것**이다. 레이어 14 의 lookup 은 40 층 중 14 번째라 스텝 시작 후 ≥ 20 ms 뒤에 오고, 레이어 1 의 lookup 도 ~1.5 ms 뒤다. GPU 는 아직 상주하지 않은 페이지를 제자리에서 fault 해 읽을 수 있다(`engram.py:1096-1103` 주석: 그래프 캡처 중엔 prefault 를 건너뛰고 "the gather then faults in place" — 이미 쓰는 경로, 문서상 ~14k rows/s).

## 3. 수정 방법 — 게이트 두 개, 기본값 행동 불변

파일: `vllm/config/engram.py`, `vllm/models/deepseek_v4_1/common/engram.py`, `vllm/models/deepseek_v4_1/nvidia/model_state.py`, `deploy/gb10-cluster/dsv41/{dsv41.env,serve-node.sh,dsv41_ctl.sh}`, `tests/kernels/test_engram.py`. 스케줄러·샘플러 불변.

### 3-1. `mmap_decode_async` (EngramConfig, int, 기본 0) — 노브 `ENGRAM_DECODE_ASYNC`

디코드 전용 스텝(`input_batch.has_prefill == False`)에서:

- `0`: 지금과 같다(표마다 동기 prefault).
- `1` **(1차 후보, 위험 0)**: 첫 표(레이어 1)만 동기로 기다리고, 나머지 표(레이어 14)는 배경 스레드에 넘기고 바로 포워드로 간다. 배경 제출을 먼저, 동기 populate 를 그 뒤에 해 두 표의 디스크 읽기가 겹치게 한다. 기대 유휴 12 → ~6 ms(레이어 14 의 20 ms 여유 안에 5~6 ms populate 가 끝난다; NVMe 가 바쁘면 남은 페이지를 GPU 가 제자리 fault, 정확성 불변).
- `2` (2차 후보): 모든 표를 배경으로. 레이어 1 의 lookup 이 콜드 페이지를 만나면 GPU fault 경로(캡처 때와 같은 경로)로 읽는다. 기대 유휴 ~1.5 ms + fault 벌점(미지수, 실측으로 판정).

구현 요지:

- `MmapEngramTable.__init__`: `decode_async` 워커(`ThreadPoolExecutor(1, "engram-async")`)와 `_pending_async` 추가. `release_after_steps > 0` 이면 비동기를 쓰지 않고 동기로 되돌린다(링 해제가 배경 populate 와 경합하지 않도록; 프로덕션은 0). 부팅 시 한 번 로그.
- `MmapEngramTable.prefault(rows, wait=True)`: `wait=False` 면 `_pages_of` → 직전 `_pending_async` drain(수십 ms 전에 끝났을 것) → `_populate(pages)` 를 워커에 제출하고 즉시 반환. `_prefetched`/`_pending_prefetch`(프리필 프리페치) 와 링은 건드리지 않는다. 프리필·혼합 배치는 기존 동기 경로 그대로.
- `ParallelEngramEmbedding.prefault(indices, wait=True)` 로 전달.
- `model_state._prefault_engram_rows`: `engram_config.mmap_decode_async` 와 `has_prefill` 로 표별 `wait` 를 정하고, 배경 표를 먼저 제출한 뒤 동기 표를 populate. 대상 순서는 `inner.layers` 순(레이어 1 → 14)이라 "첫 표 = 가장 먼저 읽히는 표"가 보장된다.

### 3-2. `mmap_min_chunk_runs` (EngramConfig, int, 기본 1) — 노브 `ENGRAM_CHUNK_RUNS`

`_run_over_pages` 의 `np.array_split(runs, min(num_threads * 4, runs))` 를 `min(num_threads * 4, ceil(runs / min_chunk_runs))` 로. 기본 1 = 현재와 바이트 단위 동일. 실험값 8: 디코드 72 run → 조각 9 개(풀 비용 −0.6~−0.9 ms/표), 프리필 800 페이지 → 100 조각(128 → 100, 사실상 동일). release/prefetch 도 같은 함수를 쓰므로 함께 적용되나 동작은 같다.

### 3-3. 측정 도구 보강 (행동 불변) — 커밋 (a) 에 포함

`_record_stats` 는 프리필과 디코드를 합친 200 회 평균(기존 r0.log 값 656~1212 rows/call)이라 디코드 스텝당 값을 주지 못한다. 행 수 < 2048 인 호출을 "decode" 버킷으로 따로 세어 200 회마다 `pages/call, ms/call` 을 찍고, 비동기 잡은 배경에서 populate 에 든 ms 를 누적해 같이 찍는다. `ENGRAM_STATS=1` 부팅 한 번으로 스텝당 prefault 시간이 r0.log 에 바로 나온다.

### 3-4. 배선

`serve-node.sh:76` 의 `--engram-config` JSON 에 `"mmap_decode_async":$ENGRAM_DECODE_ASYNC,"mmap_min_chunk_runs":$ENGRAM_CHUNK_RUNS` 추가, `dsv41.env` 에 두 항목(기본 0 / 1, 주석), `dsv41_ctl.sh:15` KNOBS 에 두 이름. 기본값이면 부팅 인자만 두 필드 늘고 동작은 같다.

커밋은 둘: (a) 게이트 배선 + 디코드 스탯 + 테스트(기본값 = 행동 불변), (b) 판정 통과 뒤 `dsv41.env` 기본값·주석·README 한 줄. 실험은 (a) 만 노드에 두고 환경 오버라이드로 켠다(#36 과 같은 방식).

## 4. 검증 방법

### 4-1. 단위 테스트 (`tests/kernels/test_engram.py` 에 추가, GPU 불필요)

- `_run_over_pages` 분할: 가짜 `work` 가 받은 조각 수·run 합을 센다 — `min_chunk_runs` 1 이면 72 run → 72 조각, 8 이면 9 조각, 64 run 이하면 인라인 1 회. 순수 numpy.
- `prefault(wait=False)`: 임시 샤드(기존 `_write_engram_shard`)로 `release_after_steps=0, decode_async=True` 표를 만들고, 호출이 즉시 돌아온 뒤 `drain` 하면 `mincore` 로 페이지가 상주하는지, 프리페치 상태(`_prefetched`, `_pending_prefetch`)가 불변인지 확인. `release_after_steps=1` 이면 동기로 되돌아가는지.
- 실행 장소: macOS 에는 `MADV_POPULATE_READ`/`libc.so.6` 이 없고 파일이 torch 를 임포트하므로 로컬 실행 불가. **헤드에서 실험 부팅의 stop→start 사이(서버 내려간 상태)** 에 `~/vllm-dsv41-venv/bin/python -m pytest tests/kernels/test_engram.py -k "mmap_table or run_over_pages" -q` 로 돌린다(CPU 전용, 수 초). 분할 산술은 로컬에서 `uv run --with numpy` 스크립트로도 먼저 본다.
- 로컬: `pre-commit run --files <변경 파일>`(ruff/mypy), `bash -n` 두 스크립트, `selftest_caps.sh`, `test_headroom.py`.

### 4-2. 클러스터 A/B (TP=4, 한 번에 한 부팅, 각 단계 전경, 사이에 확인)

공통 절차: `dsv41_ctl.sh headroom`(헤드 ≥ 5.2) → 외부 클라이언트 없음 확인(r0.log 최근 `POST /v1` 출처) → `stop` → (단위 테스트) → `ENGRAM_STATS=1 PROFILER_DIR=~/dsv41-prep/prof-i37-<tag> [오버라이드] dsv41_ctl.sh start` → r0.log 부팅 인자(두 필드 값) → health 200 → 웜업 1회.

| 부팅 | 구성 | 잰다 |
| --- | --- | --- |
| 0 `i37-base` | (a) 만, 오버라이드 없음 = 현재 기본값 | decodebench code,prose c1 gen 256 ×3, r0.log decode 버킷 ms/call, `profile_decode.sh` 의 DtoH 뒤 유휴 합/스텝. **매니저 §5-1 의 재측정**. 여기서 DtoH 뒤 유휴 ≤ 2 ms 면 코드 게이트 없이 (a) 의 스탯만 남기고 닫는다(예상: 10~12 ms, §2). |
| 1 `i37-async1` | `ENGRAM_DECODE_ASYNC=1 ENGRAM_CHUNK_RUNS=8` | 같은 디코드 3 회 + 프로파일 + `longctx.py --records 10700`(158K 니들) + casebench agent 4×6(AG4, #36 과 같은 인자) |
| 2 `i37-async2` | `ENGRAM_DECODE_ASYNC=2 ENGRAM_CHUNK_RUNS=8` | 같음. 부팅 1 이 이미 목표를 넘고 부팅 2 의 기대 이득이 프로파일상 < 2 ms 면 생략(오너 결정 §6-2). |
| 최종 | 채택 구성을 (b) 기본값으로 노드 `git am` 뒤 오버라이드 없이 `start` | decodebench c1/c4 짧은 확인, health 200, `caps` 4 대 active 1989(부하 중), 헤드 MemAvailable 기록 |

판정(매니저 §5-3 그대로): 코드 c1 스텝 ≤ 68 ms **또는** DtoH 뒤 GPU 유휴 ≤ 2 ms/스텝, 수락 길이 5.82(decode.jsonl `tokens_per_chunk`), 158K 니들 정답, AG4 총시간 208 s 대비 나빠지지 않음, 헤드 MemAvailable 최저 ≥ 3.0. 하나라도 나빠지면 미채택 → 기본값 그대로 `start` 복구. 모드 2 는 추가로 프로파일에서 engram lookup 커널(레이어 1)의 시간이 늘지 않는지(GPU fault 벌점) 본다.

노드 반영: 파이썬/셸만이라 `git format-patch | git am`(워커 3 대 `-c user.name -c user.email`, README hunk 는 `--exclude`). `.so` 재빌드 없음.

제약 준수: `nvidia-smi -pm/-lgc/-rgc`·`gpu-clock-cap.service` 손대지 않음, 임의 `rm` 없음(stop/start 의 shm 정리만), 라이브 서버 중 노드 pytest/torch 금지(stop 구간에서만), 백그라운드 체인 없음.

## 5. 이번에 하지 않을 것

- 상주 페이지 비트맵/`mincore` 건너뛰기(가설 3): 상주율 0.1 % 라 이득 0(§2-1). 구현하지 않는다.
- 인라인 임계값 상향(가설 1 처방): 콜드 순차 17 ms 로 악화(§2-2). `mmap_min_chunk_runs` 로 풀 비용만 줄인다.
- pinned 버퍼 비동기 DtoH: 이슈 본문대로 기각 유지(복사 자체는 2.7 ms/1,914 회).
- 표를 pinned RAM 에 올리는 T2W 방식(`ENGRAM_MMAP=0`): 랭크당 47 GiB, 헤드 여유 불가.
- `MADV_WILLNEED` 인라인(비차단 readahead) 변형: 모드 2 가 GPU fault 벌점으로 실패할 때의 대안으로만 기록해 둔다. 이번 A/B 에 넣지 않는다.
- 프리필·혼합 배치 경로, 프리페치(`mmap_prefetch_next_chunk`), 링(release>0) 의 동작 변경 없음.
- NVMe 교체·디스크 이전 같은 하드웨어 레버.

## 6. plan-approve 에서 정할 것

1. **모드 2(모든 표 비동기, GPU 제자리 fault) 를 프로덕션 하드웨어에서 시험해도 되는가.** 근거: 그래프 캡처 때 이미 같은 경로가 돈다(`engram.py:1096-1103`), 정확성은 페이지 fault 의미상 불변, 위험은 성능(fault 벌점)뿐. 권고: 부팅 1 이 목표(≤ 68 ms)를 넘기면 생략, 못 넘기면 시험.
2. **부팅 횟수 예산**: 최소 3(기준선·모드 1·최종), 최대 4(모드 2 포함). 각 부팅 ~2.5 분 + 측정 10~15 분(AG4 포함). 권고: 3 으로 시작.
