# 이슈 #41 계획 — 동시 2 의 +25 ms 는 두 스트림이 서로 다른 전문가를 부르는 비용이다: 프로파일 부팅 1 회로 범주별 귀속 표를 확정한다

작성 2026-09-15 19:00 KST, 워크트리 `issue-41-c2-decode-gap`(base 로컬 `main` 11d6e1629a). 매니저 조사 icmt-f5f2f407 의 순서를 따른다. 그중 1 단계(재부팅 없는 라이브 정렬 가설 검정)는 이 계획 단계에서 끝냈다. 18:42:57~18:47:15 KST 프로덕션 :8888 에 19 웨이브를 보냈고 설정 변경은 없었다. 웨이브마다 `vllm:prompt_tokens_total`·`generation_tokens_total`·`request_success_total` 증분이 프로브 요청과 정확히 같았고, r0.log 10 초 통계의 프롬프트 처리량도 프로브 합과 같아 외부 트래픽 겹침은 없다. 기존 트레이스는 워크스테이션에 복사해 집계했다(헤드 메모리 무간섭). 코드는 건드리지 않았다.

동봉: `decodebench_i41.py.txt`(라이브 프로브), `plan-probe-decode_i41.jsonl`(프로브 원본 19 줄), `steps_i41.py.txt`·`phases_i41.py.txt`(트레이스 집계).

## 1. 증상

decodebench 코드 디코드 스텝(`tokens_per_chunk / per_stream_1`)이 동시 1 68.1 ms 에서 동시 2 93.4 ms 로 +25 ms 늘어난다(i40-off r1~r3). 같은 vLLM 코어인 T2W 도 +25~26 ms, SGLang 은 +4 ms 다. 그런데 #40 켜짐 r2/r3 만 75 ms 로 이봉이다.

헤드 `~/sglang-cmp/results/decode.jsonl` 의 vLLM 계열 코드 동시 2 전부(비프로파일 12 회)를 TTFT 와 두 스트림 속도 차로 다시 보면 이봉이 도착 서명과 1 : 1 로 맞는다.

| 태그 | c1 스텝 | c2 스텝 | 증가 | c2 TTFT − c1 TTFT | 두 스트림 속도 차 (평균 − 최소, tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| vllm-prod | 75.9 | 102.8 | +26.8 | +0.169 s | 1.16 |
| t2w-boot10 | 68.0 | 94.3 | +26.3 | +0.128 s | 1.93 |
| t2w-best | 65.6 | 90.4 | +24.8 | +0.200 s | 1.92 |
| full-combo | 65.9 | 86.2 | +20.3 | +0.216 s | 3.25 |
| prof-c2-noprof | — | 101.0 | — | (c2 0.523 s) | 1.28 |
| i40-off r1 / r2 / r3 | 69.4 / 67.5 / 67.3 | 93.4 / 93.1 / 93.8 | +24.0 / +25.6 / +26.4 | +0.188 / +0.188 / +0.178 s | 1.75 / 1.56 / 1.54 |
| i40-on r1 | 69.9 | 94.5 | +24.5 | +0.197 s | 1.68 |
| **i40-on r2 / r3** | 67.0 / 66.7 | **75.5 / 75.1** | **+8.5 / +8.4** | **+0.032 / +0.018 s** | **0.00 / 0.00** |
| prod-i40 | 68.4 | 93.5 | +25.1 | +0.192 s | 1.73 |

- 느린 10 회는 모두 두 번째 요청이 첫 요청과 다른 스텝에서 프리필돼 평균 TTFT 가 0.13~0.22 s 늘고, 두 스트림 속도가 1.2~3.3 tok/s 다르다.
- 빠른 2 회는 TTFT 가 c1 과 같고 두 스트림 속도가 소수 둘째 자리까지 같다. 켜짐 r1 이 94.5 ms 에 느린 서명이므로 #40 커널이 아니라 요청 도착 타이밍이 두 모드를 갈랐다.
- 산문은 켜짐 r1/r3 이 정렬 서명(TTFT +0.038/+0.001 s)인데도 98.9/97.6 ms 이고 두 스트림 속도가 다르다(0.67/1.77). 산문은 같은 설정 반복에서도 텍스트가 달라지므로(#40 결과), 같은 스텝에 들어가도 곧 다른 토큰을 낸다. 산문에는 빠른 모드가 없다.

decodebench 는 한 웨이브의 모든 스트림에 같은 프롬프트, temperature 0 을 주고 스트림마다 따로 HTTP 요청을 보낸다(`~/sglang-cmp/decodebench.py:5`, `:51`, `:95-100` Barrier).

## 2. 근본 원인

### 2-1. 빠른 모드는 두 스트림이 같은 토큰을 같은 스텝에서 디코드할 때다 (라이브 개입 실험)

`decodebench_i41.py` 는 프롬프트를 `/tokenize`(chat template, thinking off)로 미리 토큰화한다. A 는 decodebench 코드 프롬프트(101 토큰), B 는 구조가 같은 다른 코드 프롬프트(108 토큰)다. `/v1/completions` 한 요청에 `prompt: [A, A]` 또는 `[A, B]` 를 넣어 두 시퀀스가 같은 프리필 스텝에 들어가게 한다. 스텝은 모든 스트림이 도는 구간의 청크 도착 간격 중앙값이다. 조건은 gen 256, temperature 0, ignore_eos.

| 모드 | 첫 토큰 간격 | 스텝 ms (반복) | 평균 | c1 대비 |
| --- | ---: | --- | ---: | ---: |
| c1 A | — | 68.47 / 68.18 / 68.09 | 68.25 | — |
| c1 B | — | 68.97 / 69.13 | 69.05 | — |
| pair-same `[A, A]` | 0 ms × 3 | 77.5 / 76.8 / 76.9 | **77.0** | **+8.8** |
| pair-diff `[A, B]` | 0 ms × 2 | 94.8 / 95.8 | **95.3** | **+26.6** (A·B 평균 대비) |
| pair-diff `[A, B]`, 한 스텝 어긋남 | 375 ms | 92.2 | — | — |
| A 두 요청, 두 번째를 1 s 뒤에 시작 | 1,106~1,116 ms | 86.1 / 87.9 / 86.9 | 86.9 | +18.7 |
| chat `n=2` | — | HTTP 400 (이 서버는 거절) | — | — |

- 같은 스텝이라도 토큰이 다르면 95 ms, 토큰이 같으면 77 ms 다. 원인은 정렬 자체가 아니라 **토큰(=라우팅) 동일성**이다.
- 같은 프롬프트를 1 s 어긋나게 보내면 87 ms 로 중간이다. 코드 출력이 함수 단위로 반복돼 어긋난 두 위치가 토큰 일부를 공유하는 만큼이다. 스텝 시간은 두 스트림의 토큰 겹침을 단조롭게 따른다.
- 매니저 가설(따름 정리 2)을 이 형태로 좁혀 채택한다. decodebench 동시 2 는 대개 한 스텝 어긋나 서로 다른 위치를 디코드하므로 pair-diff 와 같은 93~95 ms 가 나오고, 우연히 같은 스텝에 들어가면 pair-same 과 같은 75~77 ms 가 나온다.
- 따라서 이 이슈의 +25~27 ms 는 **내용이 다른 두 번째 스트림의 비용**이다. 그중 토큰 겹침에 따라 사라지는 몫이 95.3 − 77.0 = **18.3 ms**, 두 스트림이 같은 전문가를 불러도 남는 몫이 **8.8 ms** 다.

### 2-2. 비용 경로 (코드)

1. 부팅 로그 `Using DeepGemmFP4Experts` 대로 MoE 는 `DeepGemmFP4Experts.apply`(`vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:568-623`)가 `deepgemm_moe_permute` 뒤 FC1·FC2 그룹 GEMM 두 번을 부른다.
2. `compute_aligned_M_and_alignment`(`vllm/model_executor/layers/fused_moe/deep_gemm_utils.py:28-83`)는 CPU 쪽 전문가별 토큰 수가 없으면 작업 공간을 최악으로 잡는다. 전문가 `min(M*topk, local_num_experts)` 개 × 정렬 블록(`:80-82`)이고, 동시 2 의 12 토큰 CUDA 그래프에서는 캡처 때 고정된다.
3. `deepgemm_moe_permute` 는 `expert_ids` 를 전부 −1 로 채운다(`:512-522`). `ep_scatter` 는 토큰이 라우팅된 전문가의 블록에만 전문가 번호를 쓴다(`:147-157`, 토큰 수는 `count_expert_num_tokens`, `:525-533`).
4. SM120 그룹 GEMM 커널은 생산자·소비자 루프 모두 `m_indices < 0` 블록을 건너뛴다(헤드 `~/vllm-dsv41/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh:226-235`, `:351-356`). 원문 주석: "at decode only a few are routed; processing the rest wastes a full-width GEMM tile".

→ 그래프 안에서도 커널 일은 **토큰을 받은 전문가 블록 수**에 비례하고, 한 블록 안의 토큰이 1 개인지 2 개인지는 거의 상관없다. 같은 토큰의 두 스트림은 전문가 집합이 c1 과 같다. 다른 토큰은 스텝당 라우팅 슬롯이 6 토큰 × top-6 = 36 에서 72 로 늘고, 전문가가 384 개라 겹침이 적어 블록이 거의 두 배가 된다.

물리 해석(추정, 결과 단계에서 검산): 랭크당 활성 전문가 하나의 가중치는 FC1 1280×5120/2 + FC2 5120×640/2 바이트에 UE8M0 스케일을 더해 약 5.2 MB 다. 2-3 의 MoE 호출당 시간(FC1+FC2)은 714 → 1,162 µs 로 448 µs 늘었다. c1 의 활성 전문가를 30 개 안팎(상한 36)으로 보면 전문가당 약 15 µs 이고, 5.2 MB 를 ~350 GB/s 로 읽는 시간이다. GB10 LPDDR5X 대역폭과 같은 자릿수이므로, 한계는 연산이 아니라 가중치 읽기 대역폭이라는 해석이다.

### 2-3. 기존 트레이스 재집계 (랭크 0, 정상 상태 디코드 스텝 평균)

- c1 = `~/dsv41-prep/prof-i37-async1/code`: 현재 기본값과 같은 prefault 모드 1, 25 스텝.
- c2 = `~/dsv41-prep/prof/c2-0915/code`: 09-15 11:09, #37 이전 부팅. 같은 프롬프트 Barrier 웨이브(TTFT 0.571 s = 느린 서명), 18 스텝.
- 집계 도구는 `steps_i41.py`·`phases_i41.py`, 범주는 레포 `deploy/gb10-cluster/dsv41/prof_summary.py` 의 CATS 다.

| 항목 (ms/스텝) | c1 | c2 | 증가 |
| --- | ---: | ---: | ---: |
| 스텝 창 (첫 커널 → 다음 스텝 첫 커널) | 68.4 | 101.4 | +33.0 |
| GPU busy (커널 합집합) | 64.2 | 89.4 | **+25.2** |
| ≥ 0.5 ms 유휴 (Engram 표 대기) | 2.84 | 11.19 | c2 는 #37 이전 값이라 비교 제외 |
| < 0.5 ms 잔틈 합 | 1.37 | 0.84 | −0.5 |
| MoE 그룹 GEMM | 30.07 | 49.22 | **+19.2** |
| NCCL | 7.60 | 11.25 | **+3.7** |
| 어텐션 + 인덱서 | 16.49 | 17.63 | +1.1 |
| MoE 글루 | 1.56 | 2.46 | +0.9 |
| dense GEMM | 11.46 | 11.87 | +0.4 |
| norm/mHC·elementwise·memcpy·engram·기타 | 2.58 | 2.94 | +0.4 |
| MoE 호출당 `<1280,5120>` / `<5120,640>` (42 회/스텝 동일) | 473 / 241 µs | 748 / 414 µs | ×1.58 / ×1.72 |
| NCCL 호출당 중앙값, 4 랭크 (90 회/스텝 동일) | 62~65 µs | 89~100 µs | +25~36 µs |

CUPTI correlation 으로 각 커널을 CPU 가 어디서 런치했는지 나누면 이렇다(`phases_i41.py`, `execute_context_*` 어노테이션은 `vllm/v1/worker/gpu_worker.py:1157-1171`).

| GPU 커널 시간 (ms/스텝) | c1 | c2 | 증가 |
| --- | ---: | ---: | ---: |
| 타깃 포워드 그래프 (execute_model 안 `cudaGraphLaunch` 1 회) | 62.4 | 87.2 | **+24.8** |
| execute_model 안 eager (입력 준비, 메타데이터 빌더 = #40 경로, Engram 해시) | 0.48 | 0.58 | +0.1 |
| 드래프트 그래프 (propose 안 `cudaGraphLaunch` 1 회) | 6.1 | 6.9 | +0.8 |
| execute_model 뒤 eager (logits, 거부 샘플러, 후처리, 드래프트 준비; `vllm/v1/worker/gpu/model_runner.py:1956-2112`) | 2.25 | 2.33 | +0.1 |

사전 귀속:

- +25 ms 는 호스트 갭이 아니라 **타깃 포워드 CUDA 그래프 안의 GPU 시간**이다. MoE 그룹 GEMM 이 약 19 ms(76 %), NCCL 이 약 3.7 ms, 나머지 커널이 합쳐 약 2.8 ms 다.
- 이슈 본문 후보 2(검증 뒤처리 eager)는 +0.1 ms, #40 경로가 든 입력 준비 eager 도 +0.1 ms 이고, 잔틈은 오히려 줄었다. 9/14 보고서의 "호스트 지문" 추정은 틀렸다(정정은 #42).
- 후보 1(Engram prefault)은 이 c2 트레이스가 #37 이전이라 이번 부팅에서 잰다. 토큰이 다르면 읽을 표 페이지도 두 배가 된다(`vllm/models/deepseek_v4_1/nvidia/model_state.py:109-158`, 모드 1 은 레이어 1 표만 기다린다).
- 매니저 코멘트의 MoE 호출당 값(642 → 911 µs 등)과 다른 것은 집계 구간 차이다. 여기서는 정상 상태 디코드 스텝만 셌다.

### 2-4. 기존 데이터로 못 닫는 것 (부팅이 필요한 이유)

- 두 트레이스는 부팅과 설정이 다르고, c2 가 "어긋난 같은 프롬프트" 한 종류라 pair-same / pair-diff 몫을 트레이스로 나눌 수 없다.
- NCCL +3.7 ms 가 전송량 증가(토큰 6 → 12)인지 느린 랭크 대기인지는 랭크별·스텝별로 봐야 한다. 4 랭크 모두 호출당 중앙값이 비슷하게 늘어 전송 쪽으로 추정한다.
- 현재 기본값(모드 1)에서 동시 2 의 Engram 표 1 대기 증가분.
- 산문 동시 2 가 같은 경로인지(두 스트림 텍스트가 갈라지는 시점과 함께).

## 3. 수정 방법 (실측 계획, 코드 변경 없음)

### 3-1. 범위 결정

- `vllm/`·`tests/`·`deploy/` 코드 변경은 없다. 매니저 계획의 "필요하면" 계측 게이트(활성 전문가 카운터)도 넣지 않는다. 2-1 의 개입 실험과 2-2 의 커널 코드가 메커니즘을 정하고, 귀속에 필요한 양(범주별 GPU 시간, MoE 호출당 시간)은 기존 프로파일러로 나온다. 프로파일이 2 장과 반대로 나오면(예: pair-same 과 pair-diff 의 MoE 호출당 시간이 같음) 게이트를 붙이지 않고 멈춰 보고한다.
- 부팅 2(`8fd917c8ae` cherry-pick + `DSV41_TOKEN_REQ_KERNEL=1`)는 매니저 규칙("정렬 가설 채택 시 생략")대로 생략한다. 켜짐 r2/r3 의 75 ms 는 TTFT 서명상 같은 토큰 웨이브였고, 켜짐 r1 은 94.5 ms 에 느린 서명이다.
- 부팅은 **2 회**(프로파일 1 + 오버라이드 없는 복구 1)로, 최대 3 회 안이다.

### 3-2. 헤드 도구 (레포 밖, 사본을 노트에 `.py.txt`/`.sh.txt` 로 커밋)

1. `~/dsv41-prep/decodebench_i41.py`(계획 단계에서 설치)에 세 가지를 고친다.
   - `--kind code|prose` 추가. 산문은 decodebench 산문 프롬프트 A 와 다른 설명문 프롬프트 B 를 쓴다.
   - 스트림별 텍스트 sha1 과 첫 분기 문자 위치를 기록한다.
   - 서버가 거절하는 `n2-same` 모드를 뺀다.
2. `~/dsv41-prep/profile_i41.sh <tag> <kind> <modes>` 는 `profile_one.sh` 흐름을 따른다(start_profile → 웨이브 1 회 gen 128 → stop_profile → 4 노드 트레이스 크기 안정 대기). 다른 점은 두 가지다.
   - `profile_one.sh:11` 의 `find $P -maxdepth 1 -type f -delete` 를 쓰지 않는다. 트레이스 디렉터리는 이번 부팅용 새 경로라 비어 있고, 종류마다 새로 생긴 파일만 이름을 적어 `$P/<tag>/` 로 `mv` 한다.
   - 노드에서 요약(파이썬 json 로드)을 돌리지 않는다. #37 에서 프로파일 쓰기 중 헤드가 2.81 GiB 까지 내려갔다.
3. `prof_summary_i41.py` 는 워크스테이션에서 돌리고, 레포 `prof_summary.py` 의 CATS 를 재사용한다. 스텝마다 다음을 낸다.
   - 커널 합집합 busy, < 0.5 ms 잔틈 합, ≥ 0.5 ms 유휴와 그 뒤 첫 런치 호출.
   - 범주 합, MoE 커널 모양별 호출당 평균, NCCL 합.
   - `phases_i41.py` 방식의 CPU 런치 위치 구분(타깃 그래프 / execute_model 안 eager / 드래프트 그래프 / 뒤 eager).
   - JSON 출력.

   4 랭크를 합쳐 NCCL 을 스텝별 최소 랭크(전송 몫)와 랭크별 초과분(대기 몫)으로 나눈다.

### 3-3. 순서 (한 단계씩 전경 실행, 매 단계 결과 확인)

0. **사전 점검.** 워크스테이션에서 `/opt/homebrew/bin/bash deploy/gb10-cluster/dsv41/dsv41_ctl.sh` 로 다음을 확인한다.
   - `caps`: 4 대 active / Enabled / ≤ 2000.
   - `headroom`: 헤드 ≥ 5.2 GiB.
   - 외부 트래픽: 헤드 r0.log 의 최근 `POST /v1` 출처, `/metrics` 의 `vllm:num_requests_running` 0. 외부 192.168.1.222 가 간헐적으로 쓰므로 5 분 안에 외부 요청이 있으면 기다린다.
   - 4 노드 `~/vllm-dsv41` HEAD 와 clean 여부를 기록하고, `~/dsv41-prep/prof-i41-base` 가 없는지 확인한다.
1. **부팅 1.** `stop` 뒤 `PROFILER_DIR=/home/nacyot/dsv41-prep/prof-i41-base ... start`, health 200 을 기다린다. 유닛 Environment 에 `PROFILER_DIR` 하나뿐인지, r0.log 에 `--profiler-config` 가 찍혔는지 본다. 웜업은 `casebench.py --mode solo --prefill-tokens 16000` 1 회와 `decodebench_i41.py --warmup` 이다.
2. **같은 부팅의 비프로파일 기준.** 코드는 `--modes c1,pair-same,pair-diff --reps 3`, 산문은 `--kind prose --modes c1,pair-same,pair-diff --reps 2`. 코드가 계획 단계 값(c1 68.3 ± 1.5, pair-same 77.0 ± 2, pair-diff 95.3 ± 2 ms)을 벗어나면 멈추고 원인부터 본다.
3. **프로파일 4 종.** 각각 `profile_i41.sh` 1 회이고, 사이마다 헤드 MemAvailable 을 확인한다. 2.8 GiB 미만이면 프로파일을 멈추고 5 단계로 간다.
   - K1 코드 c1
   - K2 코드 pair-same
   - K3 코드 pair-diff
   - K4 산문 pair-same: 같은 스텝·같은 프롬프트에서도 분기하는지 텍스트로 확인한다.
4. **트레이스 복사.** 16 개를 워크스테이션의 새 디렉터리(`mktemp -d`)로 복사한다(노드 원본은 그대로 둔다).
5. **골든룰 복구(부팅 2).** `stop` 뒤 오버라이드 없이 `start`, health 200 을 기다린다. Environment 가 비었는지, `caps` 4 대(부하 중 1989)인지 본다. `decodebench.py --types code --levels 1,2 --tag prod-i41` 1 회와 `decodebench_i41.py --tag prod-i41 --modes c1,pair-diff` 1 회를 돌리고 헤드 MemAvailable 을 기록한다.
6. **집계.** 복구 뒤 워크스테이션에서 `prof_summary_i41.py` 를 돌려 `results.md` 에 귀속 표를 쓰고, 판단과 인계를 붙인다.

### 3-4. 귀속 표 (완료 기준)

열은 K1 c1 / K2 pair-same / K3 pair-diff / K4 산문 pair-same 이고, 증가분 세 개(Δsame = K2 − K1, Δdiff = K3 − K1, 라우팅 몫 = K3 − K2)를 붙인다. 각 칸에는 랭크 0 값과 4 랭크 범위를 적는다.

| 범주 | 측정 정의 |
| --- | --- |
| MoE 그룹 GEMM (타깃 / 드래프트) | `sm120_fp8_fp4_gemm_1d1d` 커널 합, 모양별 호출당 µs |
| MoE 글루 | CATS `moe_glue` (permute·unpermute·topk·양자화) |
| NCCL 전송 | 스텝별 4 랭크 최소 |
| NCCL 대기 | 랭크별 − 최소 |
| 어텐션 + 인덱서 | CATS `attention` + `attn_indexer` |
| dense GEMM | CATS `dense_gemm` |
| 기타 그래프 커널 | `norm_mhc` + `elementwise` + `memcpy` + `engram` + `other` |
| 검증 뒤처리 eager | execute_model 뒤 eager 커널 (드래프트 그래프 제외) |
| 입력 준비·메타데이터 빌더 eager (#40 경로) | execute_model 안 eager 커널 |
| Engram 표 대기 | ≥ 0.5 ms 유휴 중 바로 뒤 런치가 타깃 그래프인 것 |
| 잔틈 | < 0.5 ms 유휴 합 (호스트 런치 한계 몫) |
| 검산 | 범주 합 vs 스텝 창 vs 2 단계 비프로파일 스텝 |

완료 기준: Δdiff 의 범주 합이 비프로파일 증가분과 ± 2 ms 안에서 맞고, 몫 ≥ 1 ms 인 범주마다 4 랭크 범위가 있다.

판단과 인계(매니저 4·5 단계)는 결과에 쓴다.

- 몫 ≥ 3 ms 이고 국소 수정이 가능한 항목만 별도 이슈로 제안한다. 예상은 이렇다.
    - 라우팅 몫(~18 ms)은 가중치 읽기 대역폭이라 국소 수정 대상이 아니다. EP(#16)·b12x MoE(#31) 보류 결정과 묶어 오너 판단 자료로만 적는다.
    - NCCL 전송 몫이 3 ms 를 넘으면 디코드 크기 all-reduce 경로를 후보로 올린다.
- #42 인계:
    - 동시 ≥ 2 벤치는 서로 다른 프롬프트로 정의한다. decodebench 같은 프롬프트 동시 2 는 이봉이라 쓰지 않는다.
    - SGLang +4 ms 는 TTFT 가 +0.27 s 늘었는데 두 스트림 속도 차는 0.08 tok/s 라 이 데이터로 정렬 여부를 못 가린다. 서로 다른 프롬프트로 재측정하기 전에는 목표로 쓰지 않는다.
    - 9/14 보고서 "호스트 지문" 문장을 정정한다.

## 4. 검증 방법

- 재현 경로(재부팅 없음, 약 1 분): 헤드에서 `python3 ~/dsv41-prep/decodebench_i41.py --tag <tag> --modes c1,pair-same,pair-diff --reps 3` 을 돌리면 c1 ≈ 68, pair-same ≈ 77, pair-diff ≈ 95 ms 가 나온다. 계획 단계 원본은 `plan-probe-decode_i41.jsonl`.
- validate 단계 확인 항목:
  1. 노드 원본 트레이스(`~/dsv41-prep/prof-i41-base/<tag>/`)를 다시 복사해 커밋된 `prof_summary_i41.py.txt` 로 돌리면 표와 ± 0.1 ms 로 같다.
  2. 2 단계 비프로파일 값이 계획 단계 라이브 값과 ± 2 ms 이고, 같은 모드의 프로파일 스텝 창과 비프로파일 스텝 차이가 ≤ 3 ms 다(프로파일러 부풀림 한계).
  3. 귀속 표의 증가분 합이 비프로파일 증가분과 ± 2 ms 다.
  4. 2-2 예측: K2 의 MoE 호출당 시간이 K1 의 ± 15 % 안이고, K3 는 K1 의 1.5 배 이상이다. 틀리면 결과에 그대로 적고 해석을 고친다.
  5. 프로덕션: health 200, 유닛 Environment 비어 있음, `caps` 4 대 active / Enabled / 1989, `prod-i41` 코드 c1 68 ± 2·c2 93 ± 3 ms, 헤드 MemAvailable 기록.
  6. `git diff main -- vllm tests deploy` 가 비어 있다(노트만 바뀜).

## 5. 이번에 하지 않을 것

- MoE 커널·백엔드·EP 변경과 최적화 구현. 이 이슈의 목표는 귀속이다(EP 는 #16, b12x MoE 는 #31 에서 보류 결정).
- 활성 전문가 카운터 같은 계측 게이트 코드와 부팅 2 의 #40 커널 재측정(3-1).
- `~/sglang-cmp/decodebench.py` 원본 수정, 벤치 정의 변경, 9/14 보고서 정정, SGLang 재측정 — #42 로 넘긴다.
- 트레이스·결과 파일 삭제와 `/dev/shm` 수동 정리(ctl 의 기존 stop/start 절차만 쓴다).

## 6. plan-approve 에서 볼 것

오너만 정할 결정은 없다. 매니저 계획과 달라진 점만 확인받는다.

1. 1 단계(라이브 검정)를 계획 단계에서 끝냈다. 설정 무변경 19 웨이브, 약 4 분이다. 가설은 "정렬"에서 "토큰 동일성"으로 좁혀 채택했다.
2. 그 결과로 부팅 2 를 생략해 전체 부팅이 2 회(프로파일 + 복구)로 줄었다.
3. 프로파일 대상을 "같은/다른 프롬프트 동시 2"(별도 HTTP 요청)에서 한 요청 안의 `[A, A]` / `[A, B]` 로 바꿨다. 요청 도착 타이밍이 두 모드를 섞지 않게 하려는 것이다.
4. 계측 게이트 코드는 넣지 않는다.
5. 트레이스 요약을 노드가 아니라 워크스테이션에서 돌리고, 프로파일 스크립트의 `find -delete` 를 뺀다.
