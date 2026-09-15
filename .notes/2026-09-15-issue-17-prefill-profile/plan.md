# 이슈 #17 계획 — 128K 콜드 프리필 청크를 4 랭크·NCCL 호출 단위로 귀속하고, 회수 가능한 몫이 청크의 3 % 이상인 항목만 고친다

작성 2026-09-15 21:30 KST, 워크트리 `issue-17-prefill-profile`(base 로컬 `main` a336dccce1). 매니저 조사 icmt-61282d99 의 순서를 따른다. 계획 단계에서는 코드와 클러스터 설정을 건드리지 않았다. #12 의 기존 4 랭크 트레이스(`~/dsv41-prep/prof/i12b`, 노드 원본은 그대로)를 워크스테이션의 새 디렉터리로 복사해 다시 집계했고, 헤드·노드에서는 읽기 전용 조회만 했다.

동봉: `plan-explore_i17.py.txt`(계획 단계 탐색 집계, 버리는 도구), `plan-explore-i12b.txt`(그 출력).

## 1. 요약

- 지금 128K 컨텍스트의 청크 벽시계를 나눈 자료가 없다. 유일한 분해인 #12 프로파일(i12b)은 12K, 헤드·f323 2411 / 37cc·27c4 1989 MHz, 프리페치 페이지 건너뛰기(e706f2bbf1) 이전 상태다.
- i12b 를 NCCL 호출 단위로 다시 집계하니 청크당 **랭크 불균형 손실이 360~867 ms** 였다. 대부분 27c4 의 해시 DtoH 뒤 동기 프리폴트 유휴(청크당 494 ms)와 클럭 차이에 따른 어텐션 격차였고, 둘 다 그 뒤 바뀌었다. 그래서 새 프로파일이 필요하다.
- **(a) mHC 3 커널**은 4,092 토큰 호출에서 1.70 / 0.84 / 1.05 ms 로, GB10 대역폭 하한의 1.23~1.37 배다. 청크당 291 ms(창의 12.4 %)지만 하한까지 줄여도, 또는 GEMM 패스를 융합으로 없애도 회수는 66 ms(2.8 %) 이하다. 새 프로파일에서 같은 값이면 코드 변경 없이 닫는다.
- 판정 규칙은 **범주 총량이 아니라 회수 가능한 상한 ≥ 청크 창의 3 %** 다. (b)·(c) 는 128K 프로파일의 호출 단위 귀속으로 판정한다.
- 부팅은 없음 경로 2 회, 레버 경로 3 회다. 분석하는 동안 부팅 1 이 :8888 을 계속 서비스한다.

## 2. 현재 동작

### 2-1. 청크 하나의 경로 (코드)

- 스텝 어노테이션은 `execute_context_1(4092)_generation_0(0)` 이다(`vllm/v1/worker/gpu_worker.py:1157-1171`). SHORT_RESERVE(#34)는 대기 요청이 없으면 아무것도 떼지 않으므로 solo 프리필 청크는 4,092 토큰이다.
- **mHC.** 토큰이 16 개를 넘으면 `mhc_fused_post_pre_tilelang` 이 세 커널을 따로 부른다(`vllm/model_executor/kernels/mhc/tilelang.py:707-721` 분기, `:773-802` post + GEMM, `:827-848` pre).
    - `mhc_post_tilelang_kernel`: `tilelang_kernels.py:664-714`, n_thr 128, h_blk 1024.
    - DeepGEMM `sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u, …>`
    - `mhc_pre_big_fuse_with_norm_tilelang_kernel`

    16 토큰 이하는 post 와 GEMM 을 한 FMA 패스로 합친 `mhc_fused_tilelang`(`tilelang_kernels.py:560-661`)을 쓴다. 이 커널은 타일마다 잔차 패스를 다시 돌기 때문에 큰 토큰 수에는 맞지 않는다. 커널 파일은 업스트림 e47aa780bc, T2W 트리와 같다(매니저 확인).
- **Engram.** 표 2 개가 청크마다 조회된다.
    - 러너가 `prepare_inputs` 에서 해시를 `.cpu()` 로 가져와 이번 청크 페이지를 prefault 한다(`vllm/models/deepseek_v4_1/nvidia/model_state.py:129-160`). 이때 이번 청크를 덮는 프리페치 작업이 끝날 때까지 drain 하고, 프리페치가 채운 페이지는 건너뛴다(`common/engram.py:933-976`).
    - 다음 청크는 해시한 뒤 배경 스레드 1 개와 32 스레드 풀(`ENGRAM_THREADS`, `engram.py:731`, `:886-912`)에 populate 를 맡긴다(`model_state.py:162-214`).
    - 조회는 mmap 표를 ATS 로 직접 읽는 Triton 커널이다(`engram.py:595-653`, 그리드 `:1184-1216`).
    - 해시 열은 랭크별로 나뉜다(`:1053-1059`). 그래서 같은 청크에서 랭크가 만지는 페이지 수가 다르다. 매니저 15:56 로그의 표 하나당 값은 head 10,626 / f323 19,221 / 37cc 26,580 / 27c4 32,806 이다.
- **NCCL.** TP all-reduce 가 서브레이어마다 있고, #38 뒤에는 인덱서 층마다 all-gather 가 더해졌다. i12b 에서는 청크당 92 회였고, 4 랭크의 호출 이름과 순서가 같았다.

### 2-2. 지금 수치

S128(casebench solo 128K) 1,836 tok/s(#38 켜짐 r2·r3, 1,828 / 1,844)이다. #34 뒤 오래 뜬 프로덕션의 1 회는 1,792 였다. 128K 평균 청크 벽시계는 4,092 / 1,836 ≈ 2.23 s 이고, 3 % 는 67 ms 다. 100K 이후 청크는 어텐션·인덱서 몫 때문에 이보다 길 것으로 본다. 판정 기준 창은 새 프로파일에서 잰다(4-5).

## 3. 문제점 — 기존 분해는 지금 상태를 설명하지 못한다 (i12b 재집계)

i12b 는 09-12 09:41 KST 트레이스다. 프리페치(2a28ed7c67, 09:34)는 켜져 있었고, 페이지 건너뛰기(e706f2bbf1, 10:27)는 들어가기 전이다. 12,292 토큰이고 4,092 토큰 청크 3 개다. 집계는 `plan-explore_i17.py.txt` 로 했다.

- 커널은 CUPTI correlation 으로 런치한 스텝에 묶었다.
- NCCL 은 청크 안 k 번째 호출끼리 4 랭크를 맞췄다. 호출별 최소를 전송, 랭크 − 최소를 대기로 본다.
- 청크당 불균형 손실은 Σ(호출별 최대 대기)다.

### 3-1. 청크 창의 분해 (ms/청크, 3 청크 평균)

| 항목 | r0 (6040, 2411) | r1 (f323, 2411) | r2 (37cc, 1989) | r3 (27c4, 1989) |
| --- | ---: | ---: | ---: | ---: |
| 청크 창 (첫 커널 → 다음 스텝 첫 커널) | 2,346 | 2,345 | 2,346 | 2,346 |
| 비 NCCL 커널 busy | 1,283 | 1,262 | 1,363 | 1,351 |
| NCCL 합 | 795 | 711 | 516 | 267 |
| └ 전송 (호출별 최소 합, 92 회, 네 랭크 공통) | 219~221 | ← | ← | ← |
| 해시 DtoH 뒤 유휴 (동기 프리폴트) | 257 | 359 | 455 | **506** |
| 그 밖의 유휴 ≥ 0.5 ms | 1 | 3 | 6 | **207** |
| 잔틈 < 0.5 ms | 9 | 9 | 4 | 13 |
| 어텐션 | 360 | 360 | **416** | **398** |
| mHC 3 커널 (이름으로 분리) | 291 | 293 | 293 | 291 |
| Engram 조회 | 42 | 26 | 45 | 61 |
| memcpy | **48** | 2 | 2 | 4 |
| execute_model 뒤 런치 (샘플링·드래프트) | 26 | 25 | 25 | 26 |

| 청크 | 불균형 손실 | 원인 랭크 (호출 수 / 일으킨 대기 합) | 27c4 가 원인일 때 구간 초과분 |
| --- | ---: | --- | --- |
| 0 | 867 ms | 37cc 61 / 319 ms, 27c4 19 / 1,783 ms | 734 ms = 유휴 662 + 커널 71 (Engram +47, 어텐션 +13) |
| 1 | 598 ms | 37cc 30 / 94 ms, 27c4 44 / 1,330 ms | 544 ms = 유휴 471 + 커널 73 (Engram +35, 어텐션 +24) |
| 2 | 360 ms | 37cc 38 / 98 ms, 27c4 35 / 553 ms | 303 ms = 유휴 237 + 커널 66 (어텐션 +30, Engram +22) |

- 손실의 대부분은 27c4 의 청크 첫 all-reduce 앞 유휴다. 이는 프리폴트 drain + 전 페이지 populate 이고, 페이지 건너뛰기 이전 경로다. 매니저가 본 현재 프로덕션 로그는 drain 2~8 ms, populate 0 ms 라 이 몫은 사라졌을 것으로 본다.
- 37cc 가 원인일 때의 초과분은 어텐션(+76 / +29 / +32 ms)이다. 1989 vs 2411 MHz 클럭 차이이고, 지금은 네 대가 1989 로 같다.
- Engram 조회 청크 합의 랭크 차는 35 ms(1.5 %)다. 헤드(페이지 최소)가 f323 보다 느려 페이지 수만으로는 설명되지 않는다.
- 헤드 memcpy 48 ms 는 헤드 전용 KV 호스트 티어 저장으로 추정한다. 새 프로파일에서 커널 이름으로 확인한다.
- 호스트 잔틈은 청크당 4~13 ms(< 0.6 %)라 런치 한계는 병목이 아니다.
- CATS(`prof_summary.py`)는 `sm120_tf32_hc_prenorm_gemm` 을 `dense_gemm` 으로, Engram 해시 커널을 `engram` 으로 분류한다. 새 도구는 mHC 3 커널과 `_engram_lookup_kernel` 을 이름으로 먼저 떼어 낸다.
- NCCL 전송 92 회 × 2.4 ms 는 42 MB 링 all-reduce 의 링크 이론값(1.5 × 42 MB / 25 GB/s ≈ 2.5 ms)과 같다. 레버 대상은 아니지만 창의 9.4 % 라 귀속 표에 그대로 둔다.

### 3-2. (a) mHC 커널 대 대역폭 하한 (4,092 토큰 호출, GB10 273 GB/s)

| 커널 | 청크당 호출 | 호출당 (4 랭크) | 한 호출이 옮기는 바이트 | 하한 | 배율 |
| --- | ---: | ---: | --- | ---: | ---: |
| `mhc_post_tilelang_kernel` | 83 | 1.70~1.71 ms | 잔차 읽기 168 MB + x 42 MB + 잔차 쓰기 168 MB = 377 MB | 1.38 ms | 1.23 |
| `sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u>` | 79 | 0.84~0.85 ms | 잔차 읽기 168 MB | 0.61 ms | 1.37 |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 80 | 1.05~1.06 ms | 잔차 읽기 168 MB + 층 입력 쓰기 42 MB | 0.77 ms | 1.36 |
| 청크 합 | | 291 ms (창의 12.4 %) | | 225 ms | |

- 세 커널 모두 하한의 2 배 미만이다. 매니저 기준으로 `h_blk`·스레드 튜닝 후보가 아니다.
- 세 커널을 하한까지 줄이는 상한은 66 ms 다. post 와 GEMM 을 한 패스로 합쳐 GEMM 의 잔차 읽기를 없애는 융합의 상한은 GEMM 호출 합 66 ms 이고, 실제로는 FMA 24 출력을 같은 루프에서 계산하므로 더 작다. 둘 다 2.8 % 로 3 % 미만이다.
- 매니저가 인용한 호출당 796 µs 는 16 토큰 스텝이 섞인 평균이다. 청크 스텝만 보면 1,698~1,711 µs 다.
- 헤드 클럭이 2411 → 1989 로 내려가 연산 몫이 조금 늘 수 있으므로, 128K 프로파일에서 다시 재 판정한다.

### 3-3. 기존 데이터로 못 닫는 것

- 128K 문맥의 어텐션·인덱서 몫과 그에 따른 청크 창(3 % 기준의 분모)
- 네 대 1989 MHz, 페이지 건너뛰기, #38 분할, #37·#36 기본값에서 남은 불균형 손실과 그 원천 랭크·원천 범주
- 랭크별 Engram 조회 시간, 프리페치 populate 시간(`ENGRAM_STATS`), 그 populate 가 페이지 많은 랭크의 런치 스레드를 막는지

## 4. 바꿀 것과 접근

### 4-1. 범위와 산출물

- 1 차 산출물은 `results.md` 의 청크당·랭크별 귀속 표와 세 항목 판정이다.
- 코드 변경은 판정 규칙(4-5)을 넘는 항목이 있을 때만 한다. 허용 경로는 `vllm/model_executor/kernels/mhc/`, `vllm/models/deepseek_v4_1/common/engram.py`, `deploy/gb10-cluster/dsv41/{dsv41.env,dsv41_ctl.sh,README.md}` 이다.
- 도구는 레포 코드가 아니라 노트 사본(`.py.txt` / `.sh.txt`)으로 커밋한다(#41 방식). 레버를 쓰지 않으면 `git diff main -- vllm tests deploy` 가 비어야 한다.

### 4-2. 도구

1. **`~/dsv41-prep/i17/profile_i17.sh <tag> <prefill_tokens> <delay_s>`**(헤드, 새 디렉터리). `profile_i41.sh` 흐름을 따르며 삭제는 하지 않는다.
    - 시작 전: 4 노드 `PROFILER_DIR` 최상위 파일이 0 개이고 `<tag>` 가 없는지 확인한다.
    - `delay_s` = 0: `/start_profile` 뒤 casebench solo 를 전경으로 돌린다.
    - `delay_s` > 0: casebench solo 128K 를 이 스크립트의 자식으로 띄우고, `delay_s` 뒤 `/start_profile`, 요청이 끝나면 `/stop_profile` 한다(`curl -m 600`). 요청이 끝난 뒤 멈추는 까닭은 4-3 과 7 장에 적었다.
    - 5 초마다 헤드 MemAvailable 을 찍는다. **2.8 GiB 미만이면 casebench 클라이언트를 종료**해 요청을 취소하고, 60 초 안에 3.5 GiB 이상으로 돌아오면 `/stop_profile`, 아니면 exit 3 으로 복구 단계로 간다.
    - 트레이스 크기가 4 노드에서 멈추면(10 s 간격 2 회) 새 최상위 파일만 `<PROFILER_DIR>/<tag>/` 로 `mv -n` 한다.
    - casebench 는 README 규칙대로 레포 사본과 레포 파이썬을 쓴다: `~/vllm-dsv41/.venv/bin/python ~/vllm-dsv41/deploy/gb10-cluster/dsv41/casebench.py … --out ~/sglang-cmp/results/casebench.jsonl`. `~/sglang-cmp/casebench.py` 는 `/metrics` try/except 만 다르다.
    - `serve-node.sh:130` 의 프로파일러 JSON 은 고정이다. 이 파일은 허용 경로 밖이라 `delay_iterations`/`max_iterations`(`vllm/config/profiler.py:118-126`)는 쓰지 않고 시간으로 창을 잡는다.
2. **`prof_prefill_i17.py`**(워크스테이션, stdlib, `uv run --no-project --python 3.12 python`). `prof_summary_i41.py` 에 계획 단계 탐색 도구의 호출 단위 NCCL 매칭을 합친다. 입력은 `<tag dir>` 와 `--prompt-tokens`(casebench 기록)이다. 청크(`execute_context_1(4092)`)마다 다음을 낸다.
    - 문맥 길이: 끝의 나머지 청크부터 거꾸로 센다. 프로파일 창의 첫 스텝과 나머지 청크는 뺀다.
    - 창, 커널 합집합 busy, 스트림 겹침 보정
    - 유휴: 해시 DtoH 뒤 / 그 밖 ≥ 0.5 ms(앞뒤 커널 이름 쌍) / 잔틈 < 0.5 ms
    - 이름 분리 커널: mHC 3 종의 호출 수와 호출당 µs, `_engram_lookup_kernel` 호출별 µs, memcpy 방향별 합
    - 나머지 CATS 범주, 런치 위치(execute_model 안 / 뒤)
    - NCCL 호출 매칭: 호출 수·이름이 4 랭크에서 같은지 검사한다. 호출별 전송(최소), 랭크별 대기, 원인 랭크(최소 = 마지막 도착)를 낸다. 원인 랭크의 구간(직전 NCCL 끝 → 이 호출 시작) 초과분은 가장 짧은 구간 랭크와 비교해 범주별 커널 차와 유휴 차로 나눈다. 청크당 불균형 손실 L = Σ 호출별 최대 대기.
    - 이름이 어긋나는 청크는 #41 의 스텝별 최소 방식으로 대체하고 표에 표시한다.
    - JSON 출력
3. **ENGRAM_STATS 추출.** 프로파일 요청 시간창의 `engram prefault …` / `engram prefetch …` 로그 줄(`engram.py:903-910`, `:966-976`)을 4 랭크에서 모은다. 표·랭크별 페이지, not prefetched, drain, populate, 프리페치 queued/populate ms 를 낸다.

### 4-3. 순서 (한 단계씩 전경 실행, 매 단계 결과 확인)

명령은 워크스테이션의 `B=/opt/homebrew/bin/bash; C=deploy/gb10-cluster/dsv41/dsv41_ctl.sh` 와 헤드 ssh 로 실행한다.

0. **사전 점검.**
    - `$B $C caps`: 4 대 active / Enabled / 1989. 헤드가 unreachable 이면 헤드에서 `systemctl is-active gpu-clock-cap.service` 와 `nvidia-smi --query-gpu=clocks.sm` 조회만 한다.
    - `$B $C headroom` 기록. S128 은 256K 규칙 대상이 아니지만 기록한다.
    - 4 노드 `~/vllm-dsv41` HEAD 와 dirty 0: 계획 단계 값은 6040 278ac62ea2, f323 de62cb3b8, 37cc 723d927a7, 27c4 e7ab4deae 이고 모두 0 이다.
    - `~/dsv41-prep/prof-i17` 가 4 노드 모두 없는지 확인한다.
    - 외부 트래픽: 헤드 r0.log 최근 5 분 `POST /v1` 출처와 `/metrics` `vllm:num_requests_running` 0. 요청이 있으면 기다린다.
1. **부팅 1(프로파일).** `DSV41_SHM_DRYRUN=1 $B $C stop`(정리 대상 목록만), 그 뒤 `PROFILER_DIR=/home/nacyot/dsv41-prep/prof-i17 ENGRAM_STATS=1 $B $C start`. health 200 을 기다린다.
    - 4 노드 유닛 Environment 는 이 두 항목뿐이다.
    - r0.log 에 `--profiler-config` 가 있다.
2. **웜업.** casebench `--mode solo --prefill-tokens 16000` 1 회, `--mode mixed --prefill-tokens 6000 --decodes 2` 1 회(#38 과 같다).
3. **비프로파일 기준 S128 × 4.** 태그는 `i17-base-S128r1..r4` 다. r1 은 부팅 뒤 첫 긴 콜드 프리필이라 뺀다.
    - r2~r4 는 현재 상태 기준(A/B 의 꺼짐)이자 프로파일러 부풀림 검산 기준이다. r0.log 10 초 통계에서 문맥 100K 이후 창의 프롬프트 처리량을 뽑는다.
    - 한 회차씩 tok/s 와 헤드 최저를 확인한다. r2~r4 평균이 1,781(1,836 − 3 %) 미만이면 외부 요청, earlyoom, 로그 오류부터 보고 기록한 뒤 진행한다.
4. **P128.** `profile_i17.sh p128 128000 50`.
    - 요청 시작 50 s 뒤(문맥 약 90K) 켜고, 요청이 끝난 뒤 끈다. 문맥 90K~128K 청크 약 8~9 개를 받는다. 판정은 문맥 ≥ 100K 청크로 한다.
    - 요청 완료 뒤 멈추면 트레이스 내보내기가 프리필 중간을 막지 않는다. 또 ≥ 64K 런 뒤 `EMPTY_CACHE` 반납(2.4 GiB)이 끝난 메모리 상태에서 내보낸다.
5. **P12(대조).** `profile_i17.sh p12 12300 0` 로 4,092 청크 3 개 + 나머지를 받는다. i12b 와 같은 조건이라 청크별로 비교한다.
    - 두 프로파일 사이에 헤드 MemAvailable ≥ 3.5 GiB 와 최상위 파일 0 개를 확인한다.
6. **복사와 집계.**
    - 트레이스를 워크스테이션 `mktemp -d` 로 복사한다. 노드 원본은 둔다.
    - ENGRAM_STATS 줄을 추출한다.
    - `prof_prefill_i17.py` 로 P128·P12 를 집계하고, i12b 에도 돌려 계획 단계 탐색 값과 맞는지 본다.
    - **이 동안 부팅 1 은 :8888 을 계속 서비스한다.** 기본값에 관측 노브 2 개만 더한 상태다.
7. **판정(4-5).**
    - 레버가 없으면: 부팅 2 = 오버라이드 없는 복구(4-7), 결과 기록, #42·#43 인계.
    - 레버가 있으면: 4-6 으로 간다.

### 4-4. 귀속 표 (완료 기준)

열은 r0~r3 과 네 랭크 공통 값이다. 칸은 문맥 ≥ 100K 청크 평균(ms/청크)과 창 대비 %다. P12 와 i12b 는 같은 행으로 붙인다.

| 행 | 측정 정의 |
| --- | --- |
| 청크 창 W | 첫 커널 → 다음 스텝 첫 커널 (네 랭크 같아야 함) |
| MoE 그룹 GEMM / MoE 글루 | CATS `moe_gemm` / `moe_glue` |
| 어텐션 / 인덱서 | CATS `attention` / `attn_indexer` (#38 all-gather 는 NCCL 행) |
| mHC post / GEMM / pre | 이름 분리, 호출 수 × 호출당 |
| 기타 norm | CATS `norm_mhc` 에서 mHC 3 종을 뺀 것 |
| dense GEMM | CATS `dense_gemm` 에서 hc_prenorm GEMM 을 뺀 것 |
| Engram 조회 | `_engram_lookup_kernel` 합 (호출별 µs 따로) |
| memcpy (방향별) / elementwise / 기타 | CATS |
| 스트림 겹침 보정 | busy − 커널 합 |
| NCCL 전송 | Σ 호출별 네 랭크 최소 |
| NCCL 대기 | Σ 호출별 (랭크 − 최소) |
| Engram 프리폴트 대기 | 해시 DtoH 뒤 유휴 |
| 호스트 정지 | 그 밖의 유휴 ≥ 0.5 ms (주요 커널 쌍 명시) |
| 잔틈 | 유휴 < 0.5 ms |
| 검산 | 랭크마다 행 합 = W ± 1 % |

부속 표는 네 개다.

1. mHC 호출당 대 하한(3-2 형식)
2. 랭크별 Engram: 표당 페이지, 조회 ms/호출, drain / populate / 프리페치 populate ms
3. 불균형 손실 L 과 원천: 원인 랭크별 호출 수·대기 합, 원인 구간 초과분의 범주·유휴 분해
4. 프로파일러 부풀림: 프로파일 W 대 비프로파일 10 초 창의 4,092 / 처리량

### 4-5. 판정 규칙과 레버 후보

기준 T = 0.03 × W(P128, 문맥 ≥ 100K)다. 항목마다 회수 가능한 상한 U 를 계산하고, **U ≥ T 이고 레버가 허용 경로 안일 때만** 구현한다.

| 항목 | U 의 정의 | 구현 후보 (U ≥ T 일 때만) |
| --- | --- | --- |
| (a) mHC | 하한 배율 ≥ 2 인 커널: Σ 호출 × (측정 − 하한). 모두 < 2 면 융합 상한 = GEMM 호출 합 | 배율 ≥ 2: `mhc_post` 의 `h_blk`/`n_thr`. 헤드 서버 정지 창에서 4,092 토큰 마이크로벤치 후 `tilelang.py` 에 반영. 융합은 i12b 기준 상한 2.8 % 라 새 값이 T 를 넘을 때만 검토 |
| (b) Engram | 원인 랭크 구간 초과분 중 Engram 조회 커널 차 + 그 랭크의 프리폴트 대기 차 + 프리페치 populate 시간과 겹치는 호스트 정지 차 | 조회 커널 몫이 크면 큰 호출의 조회 그리드(`engram.py:1197-1198`). 호스트 정지 몫이 크면 `ENGRAM_THREADS` 하향(env) 또는 랭크별 값(`dsv41_ctl.sh`) |
| (c) NCCL 대기 | L 을 원천별로 나눈 것: 원인 랭크의 커널 범주 차 / 프리폴트 대기 / 호스트 정지 / memcpy | 원천이 Engram 이면 (b) 레버. 헤드 전용 일(API 서버·스케줄러·KV 호스트 티어 저장)이면 허용 경로 밖이라 인계 |

- NCCL 전송, 어텐션·인덱서, MoE GEMM 은 몫이 커도 이 이슈의 레버가 아니다. 표와 #43 인계에만 쓴다.
- 레버가 없으면 코드 변경 0 과 항목별 U / T 를 근거로 닫고, #42(보고서 정정)·#43(프리필 격차)에 귀속 표를 넘긴다.

### 4-6. A/B (구현했을 때만)

1. 로컬에서 구현하고 pre-commit(ruff, mypy, typos, shellcheck)을 통과시킨다. 노드 반영은 `git format-patch`(`mktemp -d`) → 노드별 새 배치 디렉터리 → `git -c user.name -c user.email am`(README 제외), dirty 0 이다. 헤드 단위 테스트(커널이면 `tests/kernels/test_mhc_tilelang_jit.py` 와 4,092 토큰 기존 경로 대조)는 **부팅 1 정지 뒤 서버가 없는 헤드에서만** 한다.
2. 꺼짐 대조는 부팅 1 이 떠 있을 때 돌린다.
    - `casebench --mode logits --tag i17-base-LOG --prefills 3 --prefill-tokens 16000 --decodes 2`
    - 레버가 디코드와 공유하는 경로(mHC > 16 토큰, Engram 풀)면 `~/sglang-cmp/decodebench.py --types code --levels 1 --tag i17-base`
3. **부팅 2(켜짐).** 부팅 1 과 같은 관측 노브(`PROFILER_DIR`, `ENGRAM_STATS=1`)에 레버만 더한다.
    - 웜업 → S128 × 4(r1 제외) → `i17-on-LOG` → (해당 시) decodebench → P128 1 회로 겨눈 범주가 줄었는지 확인한다.
4. **판정선(모두 충족).**
    - S128 켜짐 r2~r4 평균 ≥ max(1,836, 꺼짐 r2~r4 평균 × 1.03)
    - logits `argmax_flips` 0
    - 헤드 최저 ≥ 3.0 GiB
    - decodebench 코드 c1 스텝이 꺼짐 ± 3 %
    - 커널 변경이면 단위 대조 통과
5. 채택하면 `dsv41.env` 기본값(주석에 수치), README 문단, 노드 반영(README 제외)을 한 뒤 부팅 3. 기각하면 코드는 main 에 넣지 않고 부팅 3 은 기존 기본값이다.

### 4-7. 부팅 예산과 골든룰

| 경로 | 부팅 | 비고 |
| --- | --- | --- |
| 레버 없음 | 1 프로파일 → 2 복구 | 분석 중 부팅 1 이 서비스 |
| 레버 있음 | 1 프로파일 → 2 켜짐 A/B → 3 복구(채택 기본값 또는 기존 기본값) | 최대 3 회 |

- 매 단계 끝에서 확인한다: :8888 health 200, 4 노드 `systemctl --user show dsv41-serve -p Environment`(실험 부팅은 명시한 노브만, 복구 뒤에는 비어 있음), `caps` 4 대 1989, 4 노드 트리 dirty 0, 헤드 MemAvailable 기록.
- 복구 스모크: casebench solo 16K 1 회, `decodebench.py --types code --levels 1 --tag prod-i17` 1 회.
- 중단 조건: health 실패, 랭크 종료, earlyoom 개입, 헤드 < 2.8 GiB. 즉시 멈추고 오버라이드 없는 기본값으로 복구한 뒤 기록한다.
- 금지 사항:
    - `nvidia-smi -pm/-lgc/-rgc`, `gpu-clock-cap.service` 정지·재시작
    - 서버가 떠 있는 노드에서 pytest·torch 프로세스
    - 글롭·변수 경로 `rm`(트레이스·기록 삭제 없음, shm 은 ctl 의 기존 정리만)
    - 백그라운드 체인: 스크립트 한 번은 전경 한 단위이고 약 2~3 분이다.

## 5. 영향 사이트

- **결과 소비자.** #42 보고서 정정, #43 프리필 격차 메타, 채택 시 `deploy/gb10-cluster/dsv41/README.md` 의 기본값 목록(복구 규칙 `:297-305`).
- **(a) 레버를 쓸 때.**
    - `mhc_fused_post_pre_tilelang`: `tilelang.py:622-855`. 호출 경로는 CustomOp `mhc_fused_post_pre`(`vllm/model_executor/layers/mhc.py:448-540`)이고, V4.1 디코더 층과 DSpark 드래프트가 쓴다.
    - JIT 워밍업 `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py`
    - 테스트 `tests/kernels/test_mhc_tilelang_jit.py`, `tests/kernels/test_mhc_kernels.py`
    - 동시 디코드 배치가 16 토큰을 넘을 때(SEQS 16 × SPEC_K+1)도 같은 분리 경로를 타므로 디코드 확인이 필요하다.
- **(b) 레버를 쓸 때.**
    - `MmapEngramTable` 풀과 프리페치(`engram.py:731`, `:839-912`), `EngramEmbedding.lookup` 그리드(`:1184-1216`)
    - 러너 prefault·prefetch(`model_state.py:129-214`)
    - #37 디코드 비동기 populate 가 같은 풀을 쓴다(`engram.py:914-931`).
    - 설정 `serve-node.sh:77`(`mmap_prefault_threads`), `dsv41_ctl.sh:15` KNOBS(`ENGRAM_THREADS` 이미 포함)
- **측정 인프라(변경 없음).** `serve-node.sh:43`(`ENGRAM_STATS` → `VLLM_ENGRAM_MMAP_STATS`), `:126-130`(프로파일러), `dsv41_ctl.sh:15`, `prof_summary.py` CATS(재사용)

## 6. 검증 방법

validate 단계에서 확인할 것:

1. 노드 원본 트레이스(`~/dsv41-prep/prof-i17/{p128,p12}/`)를 다시 복사해 커밋된 `prof_prefill_i17.py.txt` 로 돌리면 `results.md` 표와 ± 0.1 ms 로 같다.
2. 같은 도구를 i12b 에 돌리면 3-1·3-2 값(창 2,346 ms, 전송 219~221 ms, mHC 1,698~1,711 / 835~849 / 1,048~1,056 µs, 불균형 손실 867 / 598 / 360 ms)이 나온다.
3. P128 분석 청크마다 NCCL 호출 수·이름이 4 랭크에서 같고, 랭크마다 행 합이 W ± 1 % 다.
4. 프로파일러 부풀림: P128 의 W 와 비프로파일 S128 r2~r4 의 문맥 100K 이후 10 초 창에서 구한 청크 시간 차이가 ≤ 5 % 다. 넘으면 결과에 적고 호스트 범주 해석에 단서를 단다.
5. P12 대 i12b 차이가 알려진 변경과 같은 방향이다. 프리폴트 대기가 줄고(e706f2bbf1), 어텐션 랭크 차가 사라지고(클럭), 인덱서가 분할되는지(#38) 본다. 틀리면 그대로 적고 해석을 고친다.
6. 세 항목마다 U, T, 판정이 `results.md` 에 있다.
7. 레버를 썼으면: A/B 표가 4-6 판정선을 충족하는지와 단위 테스트 로그. 노드 4 대가 dirty 0 이고 반영 커밋 내용이 로컬과 같은지, `dsv41.env` 기본값과 README.
8. 프로덕션: health 200, 4 노드 Environment 비어 있음, `caps` 4 대 1989, 4 노드 dirty 0, 복구 스모크 결과, 헤드 MemAvailable 기록.
9. 레버가 없으면 `git diff main -- vllm tests deploy` 가 비어 있다(노트만 바뀜).

## 7. 이번에 하지 않을 것

- Engram 해시 열의 랭크 재배치. vocab 구간이 연속이어야 하는 가중치 로더(`engram.py:580-592`)까지 바뀐다(매니저 제외).
- 스케줄러와 인덱서 분할(#34, #38) 변경.
- 표에서 몫이 크게 나와도 레버 대상이 아닌 것: NCCL 설정(`NCCL_LEAN`·버퍼·채널)·전송 경로, KV 호스트 티어·API 서버 배치, 어텐션·MoE 커널. #43 으로 인계한다.
- 융합 mHC 커널 신규 작성. 상한이 T 미만이면 하지 않는다(i12b 기준 2.8 %).
- `serve-node.sh` 프로파일러 JSON 수정(`max_iterations` 등). 허용 경로 밖이고 시간 창으로 충분하다.
- 매니저 계획의 "시작 약 12 s 뒤 `/stop_profile`". 대신 요청이 끝난 뒤 멈춘다(4-3 의 4).
- 트레이스·결과 삭제, `/dev/shm` 수동 정리, 클럭·persistence 조작.

## 8. plan-approve 에서 볼 것

오너만 정할 결정은 없다. 매니저 계획에서 구체화하거나 바꾼 점만 확인받는다.

1. **판정 규칙.** "청크의 3 % 이상인 항목"을 회수 가능한 상한 U ≥ 3 % × W(문맥 ≥ 100K 청크)로 적용한다. mHC 3 커널은 총량으로 12.4 % 지만 하한의 1.2~1.4 배라 U 가 2.8 % 다.
2. **계획 단계 사전 결과.** i12b 재집계로 (a) 는 코드 변경 없이 닫힐 가능성이 크다. 128K 프로파일에서 호출당 값만 다시 확인한다.
3. **NCCL 대기를 호출 단위로 가른다.** 청크당 불균형 손실과 원인 랭크, 원인 구간의 범주·유휴 분해를 낸다(#41 의 스텝별 최소보다 세분). i12b 에서 92 회 호출 이름이 4 랭크 모두 맞았다.
4. **프로파일 창.** P128 은 요청 50 s 뒤 켜고 요청 완료 뒤 끈다. 부팅 1 에 `ENGRAM_STATS=1` 을 더한다.
5. **분석 중 서비스.** 부팅 1(기본값 + 관측 노브 2 개)이 분석하는 동안 :8888 을 계속 서비스한다. 다운타임을 줄이고 부팅 3 회 예산을 지키기 위해서다. 단계 끝은 Environment 가 빈 복구 상태다.
6. **A/B 대조.** 켜짐 부팅도 같은 관측 노브를 달아 꺼짐(부팅 1 기준)과 레버만 다르게 한다. 켜짐 부팅에서 P128 을 한 번 더 받아 겨눈 범주가 줄었는지 확인한다.
