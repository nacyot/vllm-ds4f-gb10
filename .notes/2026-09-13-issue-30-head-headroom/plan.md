# 이슈 #30 계획 — head 신규 부팅 유휴 여유 편차(4.6~7.1 GiB)의 원인 확정 + 493K 콜드 프리필 바닥 게이트

작성 2026-09-13 00:5x KST, 워크트리 `issue-30-head-headroom`(base `main` 8c84efe16e). 매니저 레시피 icmt-dbfbd3cb 를 따르되, plan 단계의 읽기 전용 실측으로 원인은 이미 상당 부분 확정됐다. 아래 §2 가 그 근거다.

## 1. 증상

채택 구성 신규 부팅 + 82K 웜업 뒤 head(gx10-6040) 유휴 MemAvailable 이 어떤 부팅은 7.1 GiB(#25 S4), 어떤 부팅은 4.6 GiB(#19)라 493K 콜드 프리필 가드(≥ 5.2)를 통과할지 예측할 수 없다. #27 이 남긴 "493K 콜드 프리필 바닥 ≥ 3.0 GiB" 게이트가 미완이다.

## 2. 진단 (전부 읽기 전용: /proc, journal, sar, r0.log, #25 워커 타임라인, 홈랩 VictoriaMetrics 15 s 시계열)

### 2-1. 프로세스 anon 은 자라지 않았다 — swap 배치 차이 (매니저 실측 재확인)

지금(#20 부팅, 8K 프로브 뒤) head 3 프로세스:

| 프로세스 | pid | RssAnon | RssFile | RssShmem | VmSwap |
| --- | ---: | ---: | ---: | ---: | ---: |
| VLLM::Worker_TP0 | 3597242 | 2,836 MiB | 862 | 3,480 | 0 |
| VLLM::EngineCore | 3597084 | 808 | 392 | 3,072 | 0 |
| API 서버(`vllm serve`, EngineCore 의 부모) | 3596877 | ~870 | | | 0 |

S4 시작(19:34:13)의 worker 는 RssAnon 2,172 + VmSwap 665 = 2,837 → 지금과 같다. 세 프로세스 콜드 anon 약 1.7 GiB 가 **swap 으로 나가 있느냐** 가 4.6 ↔ 7.1 의 전부다. `SUnreclaim` 2.1~2.2 GiB 는 4 노드 공통 기준값(매니저).

### 2-2. swap-out 은 점진적이 아니라 특정 창의 "버스트" 다 (sar, 10 분 간격, 09-12)

| 창(KST) | 해당 부팅/작업 | kbswpused 변화 | pswpout/s | pgscan_kswapd/s | pgsteal/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 17:07~18:08 부팅 5회(각 웜업) | — | 0 | ≤ 1.4 | 644~1,266 | 1,277~2,498 |
| 18:20~18:30 | 18:18 부팅 + 웜업 + 493K 콜드 프리필(18:23~18:29) | +1,174 MiB | 474 | 2,392 | 4,303 |
| 19:06~19:22 b1~b4(복원·bench2) | — | 0 | ≤ 1.4 | 644~1,266 | 1,277~2,498 |
| 19:30~19:40 | **b5 부팅 + 웜업 + S4 시작** | **+2,325 MiB** | 946 | 3,179 | 4,704 |
| 21:50~22:00 | #27 b2(21:48 부팅) + 웜업 | +2.3 GiB(정지로 해제) | 1,001 | 2,383 | 3,272 |
| 23:00~23:10 | **#19 부팅(22:58) + 웜업** | 0 | 0 | **0** | 0 |
| 23:40~23:50 | **#20 부팅(23:43) + 8K** | 0 | 0 | 0 | 0 |

### 2-3. b5 타임라인 (레시피 1 완료) — 15 s 시계열로 swap-out 시각을 웜업 안으로 특정

출처: `journalctl --user -u dsv41-serve`, `dsv41-r0.log`, #25 워커 타임라인(tsk_9ddd0518), VictoriaMetrics `gb10_swap_used_bytes / gb10_mem_available_bytes{node="gx10-6040"}` (head 의 node-metrics-exporter :9110, 15 s).

| KST | 사건 | head MemAvailable | swap 사용 |
| --- | --- | ---: | ---: |
| 19:22:06 | b4 부팅(채택 구성) | | 141 MiB |
| 19:24:21~19:25:15 | b4 82K 웜업(TTFT 53.9 s), 최저 3,340 MiB, swap-out 없음 | 6,194 → 3,340 → 5,430 | 141 |
| 19:26:25~19:28 | `i25-bench2.service` C1/C4(완료, i25f.json 기록) | ~4.7 GiB | 142 |
| 19:29:21 | `dsv41_ctl.sh stop` | 117 GiB | 141 |
| 19:30:21 | b5 부팅 시작(drop_caches → instanttensor 스트림, 로딩 중 20.5 GiB 유지) | | 141 |
| 19:32:16 / 19:32:41 / 19:32:45 | 모델 로드 완료 / init engine(그래프 0.86 GiB) / startup complete — 부팅 인자·KV 4,571,060 토큰·그래프 크기 모두 #19 부팅과 동일(diff 0) | 6,372 (19:32:45) | 141 |
| 19:32:58 | 82K 웜업 시작(Triton JIT 4개 19:32:59~19:33:00, 다른 부팅과 동일) | 5,549 (19:33:00) | 141 |
| 19:33:15 / :30 / :45 | 웜업 프리필 진행 | 4,381 / 3,959 / 3,880 | 141 / 148 / 148 |
| **19:33:45~19:34:00** | **swap-out 1,747 MiB** + `empty_cache … 2.93 GiB released`(19:34:00), TTFT 63.5 s(다른 부팅 51~54 s) | 5,171 → 7,110 (19:34:15) | **1,895** |
| 19:34:13 | memlog 시작: MemFree 5,131, Cached 4,153(Shmem 3,454 → 파일 캐시 0.7 GiB), AnonPages 3,020, worker VmSwap 665 | 7,114 | 1,886 |
| 19:34:23~19:41 | S4 493K 콜드 프리필(게이트 6.87), 첫 1 분에 +0.5 GiB 추가 swap-out, 바닥 4.82 | 4,990~5,270 → 7,516 | 2,414 |

b5 와 S4 사이에는 웜업 외에 아무것도 돌지 않았다(bench2 는 b4 에서 끝남, 오너 `uvm-stall-sentinel` 은 이 창에 로그 없음, `kv-prune` 은 파일 삭제만). **swap-out 을 일으킨 "선행 작업" 은 별도 작업이 아니라 b5 자신의 82K 웜업 프리필이다.** 같은 패턴이 #27 b2(21:48 부팅)에서도 재현돼 있다: 웜업 21:51:13 → 21:51:45 swap 506 → 21:52:15 **2,637 MiB**, 뒤 유휴 7.3~7.8 GiB(#27 results "head 유휴 7.66").

### 2-4. 대조: 여유가 더 적었던 #19 웜업은 swap 0

| 부팅 | 부팅 직후 MemAvailable | 웜업 중 최저(15 s 샘플) | swap-out | 웜업 뒤 유휴 |
| --- | ---: | ---: | ---: | ---: |
| b4 19:22 | 6,194 | 3,340 | 0 | 5,430 → 4.7 |
| b5 19:30 | 6,372 | 3,880 (직후 swap) | +1,747 | 7,110 |
| #27 b2 21:48 | 6,921 | 4,283 (직후 swap) | +2,496 | 7,850 → 7,3xx |
| #19 22:58 | 5,109 | **2,988** | **0** | 4,716 |
| #20 23:43 | 5,076 | (8K 만) | 0 | 4,053 |

즉 swap-out 은 MemAvailable 부족(저수위 워터마크 low 534 MiB / min 410 MiB 근처)으로 설명되지 않는다. #19 는 2.99 GiB 까지 내려가고도 회수 스캔 자체가 0 이었고, b5·#27 b2 는 3.9~4.3 GiB 에서 1.7~2.5 GiB 를 내보냈다.

### 2-5. 메커니즘: 웜업의 2.93 GiB 재성장이 요구하는 고차(2 MiB) 페이지 vs 단편화된 free 풀

- 웜업 82K 프리필은 torch 할당자를 2.93 GiB 키우고(`expandable_segments`, cuMemCreate 2 MiB 단위, GB10 통합 메모리라 **물리적으로 연속한 2 MiB = order-9 페이지**가 필요) 끝나면 `EMPTY_CACHE_MIN_TOKENS=65536` 규칙으로 되돌려 준다(`2.93 GiB released`).
- head 의 free 풀은 상시 극단적으로 단편화돼 있다. 지금(`/proc/buddyinfo` Normal): order 9 = 5, 10 = 2, 11 = 1, 12 = 0, 13 = 5 블록. free 2.55 GiB 가 거의 전부 저차 조각이다. `/proc/vmstat`: compact_stall 436,708 중 compact_fail 386,221(88 %), allocstall_normal 4.0 M, pgmigrate_success 702 M. 오너의 `uvm-stall-sentinel.service`(매분, "UVM fragmentation stall — PSI full ≥ 50 & order≥9 = 0 이면 drop_caches + compact_memory")가 존재하는 이유가 이것이며 09-01 이후 32회 개입했다(마지막 09-12 00:16; 09-12 저녁 창에는 PSI 가 50 에 못 미쳐 개입 없음).
- order-9 블록이 없는 상태에서 2 MiB 청크를 연속 요구하면 커널은 kswapd(order-9 워터마크)·직접 컴팩션으로 대상 pageblock 을 비운다. 이때 이동 불가한 page cache 는 회수되고, LRU 의 **콜드 anon(API 서버·EngineCore·worker 초기화 힙, swappiness 60)** 이 swap 으로 나간다. 회수는 고차 워터마크를 맞출 때까지 과잉 진행되므로 끝나면 MemFree 가 크게 남는다 — S4 시작 시 MemFree 5.1 GiB, 파일 캐시 0.7 GiB(지금 2.4 GiB)가 그 지문이다. 웜업 TTFT +10 s(63 vs 51~54 s)는 그 스톨이다.
- 반대로 free 풀에 order-9 블록이 충분하거나 청크 요구가 기존 블록으로 충족되면(#19·#20) 스캔 0, anon 잔류 → 4.6~4.9.
- 493K 콜드 프리필도 같은 경로를 밟는다: S4 첫 1 분 +0.5 GiB, 18:18 부팅의 493K 중 +1.15 GiB, #27 동시 복원 창 +2.3 GiB. 즉 "493K 바닥" 도 프리필 시작 시점의 단편화 상태에 좌우된다(S4 바닥 4.82 는 anon 이 이미 빠진 상태의 값).

**근본 원인(확정 수준)**: 신규 부팅 여유 편차 4.6~7.1 GiB 는 코드·부팅 인자·프로세스 크기 차이가 아니라, **첫 긴 프리필(웜업)의 2 MiB 단위 할당자 재성장이 단편화된 head free 풀에서 컴팩션·회수를 촉발해 콜드 anon ~1.7 GiB 를 swap 으로 내보내고 파일 캐시를 비웠는가(→ 7.1) 아닌가(→ 4.6)** 로 결정된다. 촉발 여부는 그 순간의 order≥9 블록 수(부팅 이력·churn 에 따른 단편화 상태)에 달려 있어 부팅마다 다르다. 남은 미확정 하나: 촉발 순간의 order≥9 수와 컴팩션 카운터 증분을 1 s 로 직접 본 기록이 없다 — 레시피 2 의 부팅 1회가 이를 채운다(§3-2).

부수 발견: `memlog.py` 는 Engram 샤드 2개를 `MAP_PRIVATE|PROT_WRITE` 로 매핑해 실행 중 Committed_AS 를 +189 GB 부풀린다(sar 19:40 kbcommit 207 GB, %commit 150). `overcommit_memory=0` 이라 무해하지만 모니터링을 오염시키므로 이번에 `PROT_READ` 로 고친다(§3-1).

## 3. 작업 계획 (레시피 0~6)

### 3-1. memlog.py 확장 (코드, 작음) — `deploy/gb10-cluster/dsv41/memlog.py`

- `head_pids()`: 기존 `pgrep -f VLLM::Worker`(앵커 `^VLLM::Worker` 로) 로 worker 를 잡고 `/proc/<pid>/stat` 의 ppid 를 두 번 따라가 EngineCore → API 서버를 얻는다(현 트리 3597242 → 3597084 → 3596877, comm `vllm`). 워커 노드에서는 부모가 랭크 런처라도 무해.
- 컬럼: 기존 컬럼 순서·이름 유지(이전 CSV 와 비교 가능) + `SUnreclaim` 을 MEMINFO 끝에, `/proc/vmstat` 카운터 `pswpin, pswpout, pgscan_kswapd, pgscan_direct, compact_stall, compact_success, allocstall_normal`(누적값, 증분은 후처리), `/proc/buddyinfo` Normal 존 order≥9 합 `free_order9plus`, 3 프로세스 `e_RssAnon,e_VmSwap,a_RssAnon,a_VmSwap`(worker 는 기존 `RssAnon/VmSwap` 그대로) 을 뒤에 붙인다. 레시피는 pswpin/pswpout 만 요구하지만 §2-5 의 메커니즘을 같은 CSV 에서 판정하려면 컴팩션·스캔·order≥9 가 필요하다(코드는 튜플에 이름 추가 수준).
- `Shard`: `libc.mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0)` 로 바꿔 Committed_AS 부풀림 제거(mincore 는 주소만 필요).
- torch 미사용 유지(ctypes·mmap·subprocess·os). 파싱을 순수 함수(`parse_status(text)`, `parse_vmstat(text)`, `parse_buddyinfo(text)`)로 분리.
- 테스트 `deploy/gb10-cluster/dsv41/test_memlog.py` 1 파일: 위 세 파서와 `head_pids` 의 ppid 체인(가짜 `/proc` 루트 인자)과 "import 후 `torch` 가 sys.modules 에 없음" 을 검사. 워크트리에 `uv venv --python 3.12 .venv && uv pip install pytest ruff` 로 실행(메인 체크아웃 .venv 는 손대지 않음). `ruff check/format` 통과.
- head 반영: 커밋 → `git format-patch -1 -o $(mktemp -d)` → head `~/vllm-dsv41` 에서 `git am`(head 의 memlog.py 는 워크트리와 md5 동일 79a67009…, 트리 clean 이라 적용 예상). 실패하면 #20 처럼 squash sync 패치, 그것도 안 되면 `~/dsv41-prep/bench/i30/memlog.py` 새 디렉터리 사본으로 실행하고 기록. `~/dsv41-prep/memlog.py` 심링크(#28)는 그대로.

### 3-2. 신규 부팅 1회 + 웜업, memlog 켜고 (레시피 2 = 레시피 3 의 재현을 겸함)

순서(각 단계 포그라운드, 30~60 s 관찰, 이상 시 중단):

1. `dsv41_ctl.sh caps` / `status` / `headroom`(기록용) → `dsv41_ctl.sh stop` → `dsv41_ctl.sh shm`(4 노드 잔재 0 확인).
2. head: `systemd-run --user --unit=i30-memlog ~/vllm-dsv41/.venv/bin/python ~/vllm-dsv41/deploy/gb10-cluster/dsv41/memlog.py ~/dsv41-prep/bench/i30-boot-memlog.csv 1`(#25 와 같은 측정 전용 유닛, 서버 부팅 전에 시작해 부팅 자체를 담는다) + 시작 시각의 `/proc/vmstat` pswpout·compact_stall, `/proc/buddyinfo` 스냅샷.
3. `dsv41_ctl.sh start`(캡 게이트) → health 200 까지 30 s 폴링(예상 134~159 s).
4. 부팅 직후 스냅샷 A: 3 프로세스 RssAnon/VmSwap, MemAvailable/MemFree/Cached/Shmem, SwapFree, pswpout·compact_stall 증분, order≥9 수, `systemctl --user show dsv41-serve -p MemorySwapPeak`.
5. 82K 웜업 1회 `kvoff_probe.py i30-warm W30 5600`(head, `~/vllm-dsv41/.venv/bin/python`, 정답 "12", TTFT 기록) → 스냅샷 B(같은 항목) + memlog 에서 웜업 창의 MemFree 최저·pswpout 증분·compact_stall 증분·order≥9 궤적.
6. 판정: (a) 웜업 중 pswpout 증분 ≥ 수십만 페이지·VmSwap > 0 → §2-5 를 라이브로 확정(7.1 급). (b) 증분 0 → "부팅+웜업만으로는 swap-out 없음(4.6 급)" 이 이번 실측이고, 그때의 order≥9 수·compact_stall 증분이 "왜 안 났는가" 의 수치다. 어느 쪽이든 1회로 끝내며, 7.1 상태를 "낚기" 위한 추가 부팅은 하지 않는다.

### 3-3. 493K 콜드 프리필 게이트 (#27 후속, 레시피 4)

- 웜업 뒤 `dsv41_ctl.sh headroom` ≥ 5.2 **일 때만**: 새 3글자 salt(S4·S27a·Wqz·W30 등 기사용 제외), `kvoff_probe.py i30-493k <salt> 33000`(493,015 토큰)을 `systemd-run --user --unit=i30-493k` 로 띄우고 30 s 마다 head MemAvailable·health·earlyoom·유닛 상태를 포그라운드로 확인(#25/#19 방식; 프로브 자체의 2 s 모니터가 2.8 GiB 에서 중단). 기록: memlog 1 s 최저, `mem_avail_min_gib`, 정답, 소요(예상 380~400 s), earlyoom 0, `empty_cache` 반납 로그, 프리필 중 pswpout·compact_stall 증분, 뒤이어 8K 프로브 1회(`kvoff_probe.py i30-8k <salt2> 700`) 와 종료 head 값. 바닥 ≥ 3.0 이면 README 표에 행 추가.
- 5.2 미만이면 **돌리지 않고** README 표에 "Not run" 행: 시작값, 예상 바닥(시작 − 2.1~2.2), 그리고 §2-5 근거(4.6 급 = anon 미방출 상태). 2.8 중단은 시험하지 않는다.

### 3-4. 오너 레버 제안 — 별도 이슈 1건(`ryno issue add`, 라벨 `decision`, 구현 금지)

수치(§2)와 함께 선택지를 나열만 한다:

- (a) 콜드 anon 을 긴 프리필 전에 의도적으로 내보내기: `vm.swappiness` 상향 또는 serve 유닛 `MemoryHigh=` 일시 적용. 효과: 7.1 급을 결정적으로 만든다. 위험: 프리필 중 throttle, 프리필 뒤 swap-in 지연(첫 긴 프리필 2~15 s 지연은 이미 dsv41.env 에 기록됨).
- (b) 단편화 해소를 웜업 전에 강제: 오너 sentinel 의 `compact_memory`(+drop_caches) 를 부팅 직후 1회 호출하는 훅. 효과: 2 MiB 청크 스톨 자체를 줄여 swap-out 도 줄 수 있음(그러면 4.6 급이 기본이 됨 — 493K 가드는 여전히 5.2). `vm.compaction_proactiveness` 는 이미 100.
- (c) 재성장 자체를 없애기: `EMPTY_CACHE=0`(#12/#15 에서 메모리 사유로 기각된 트레이드오프 — 493K 뒤 head ~3 GiB) 또는 `EMPTY_CACHE_MIN_TOKENS` 를 웜업 82K 보다 크게 두어 웜업 세그먼트를 유지(긴 프리필은 어차피 재성장). 수치만 적고 권고 안 함.
- (d) 현상 유지 + "신규 부팅에서 493K 콜드 프리필 보장 없음" 문서화(이번 README 반영으로 이미 충족).
sysctl·유닛 파일·sentinel 은 이 이슈에서 바꾸지 않는다.

### 3-5. README `Long cold prefill headroom` 절

"Fresh-boot headroom varies from 4.6 to 7.1 GiB" 문단을 원인(§2-5 요약: 웜업 재성장 → 단편화 컴팩션 → 콜드 anon swap-out 여부)과 3 프로세스 표(부팅 직후·웜업 뒤·493K 뒤)로 바꾸고, sar/15 s 버스트 표를 축약해 넣고, 표에 #30 행(실측 또는 Not run)을 추가한다. SUnreclaim 은 4 노드 기준값 한 줄. `memlog.py` 새 컬럼 설명 갱신.

### 3-6. 종료 상태

채택 구성은 3-2 의 부팅 그 자체(추가 재기동 없음). 4 대 serve active, 캡 active·1989, health 200, head MemAvailable 기록(짧은 프로브 뒤 3.4~4.0 정상), 작업 창 earlyoom 0, i30-memlog 유닛 정지, :8889 그대로.

## 4. 검증 방법

- 로컬: `.venv/bin/python -m pytest deploy/gb10-cluster/dsv41/test_memlog.py -v`(+ 기존 `test_headroom.py` 회귀 확인), `ruff check`/`ruff format --check` 대상 파일, `git diff --stat` 이 `deploy/gb10-cluster/dsv41/{memlog.py,test_memlog.py,README.md}` 와 `.notes/…/{plan,results}.md` 로만 한정.
- head 스모크(서버 살아 있는 상태, torch 없음): `~/vllm-dsv41/.venv/bin/python -c "import memlog"` 및 5 s 짧은 실행으로 헤더 컬럼·3 pid·order≥9 값이 채워지는지 확인 — 3-2 의 stop 전에 수행.
- 클러스터: 3-2/3-3 의 스냅샷 A/B(+C) 표와 memlog CSV(`~/dsv41-prep/bench/i30-*.csv`), `journalctl -u earlyoom` 작업 창 0 건, `dsv41_ctl.sh status`·`caps` 종료 출력. results.md 에 시각(KST)과 함께 기록.

## 5. 하지 않을 것

- sysctl(swappiness·drop_caches·compact_memory)·cgroup 한도·swapoff·sentinel/kv-prune 유닛·gpu-clock-cap·nvidia-smi 변경: 전부 오너 레버(3-4 에 제안만).
- 493K 를 5.2 미만·웜 서버에서 시작, 2.8 중단 시험, 7.1 상태를 얻기 위한 반복 부팅.
- `vllm/` 코드 변경(#24·#27 코드는 원인이 아님), kvfs 스토어 삭제, `~/dsv41-prep` 기존 파일 교체(memlog 는 git am 또는 새 디렉터리).
- 단편화의 "원인의 원인"(어느 boot churn 이 order≥9 를 소진하는가) 추적: 이번 범위 밖, 결정 이슈에 관찰치만 남긴다.

## 6. plan-approve 에서 확인받을 것 (오너/매니저 결정)

1. 레시피 2 의 부팅 1회에서 headroom < 5.2 가 나오면(#19·#20 재현, 확률 높음) 493K 는 "Not run" 행으로 마감한다 — 레시피 4 그대로. 이의 없으면 그대로 진행.
2. memlog 추가 컬럼을 레시피(pswpin/pswpout)보다 넓힌 것(pgscan·compact·allocstall·order≥9)과 `PROT_READ` 수정: 같은 파일·소규모라 포함했다. 축소를 원하면 컬럼만 빼면 된다.
3. 결정 이슈 3-4 의 선택지 (a)~(d) 구성.
