# 이슈 #38 계획 — T2W 인덱서 프리필 TP 분할(`DSV41_INDEXER_TP_SPLIT`) 이식과 꺼짐/켜짐 실측

작성 2026-09-15 KST, 워크트리 `issue-38-indexer-tp-split`(base `main` c8cb661ceb). 매니저 코멘트 icmt-558a9282 의 레시피·범위·제약을 따른다. 코드는 아직 건드리지 않았다. 조사는 읽기 전용이다: 로컬 파일, 헤드 `~/t2w-trees/t2w-best` 의 `git diff`, max10 초안 테스트 파일, 헤드 `~/sglang-cmp/*.py`, 모델 `config.json`.

## 1. 현재 동작과 문제

- `vllm/model_executor/layers/sparse_attn_indexer.py:469-572` 의 프리필 청크 루프는 모든 TP 랭크가 같은 일을 한다. 인덱서가 ReplicatedLinear 라서, 청크의 모든 행을 인덱스 K 전체와 대조해 점수를 매기고(`fp8_fp4_mqa_logits`) 행별 top-k 를 구한다. 후보 원천 층이면 `_select_candidate_blocks`, 그 뒤 층이면 `_apply_candidate_mask` 도 돈다.
- 모델 설정: `index_source_layer_ids` 8개(2,8,14,20,24,28,32,36), `index_topk` 512, `candidate_source_layer_id` 20(`candidate_topk_blocks` 2048, 블록 8). 점수 비용은 행 수 × 누적 K 로 늘어 128K 에서 커진다. 12K 프로파일의 0.5% 는 짧은 컨텍스트 값이다(매니저 §1).
- 128K 콜드 프리필: #36 노브 셋 1,464 tok/s. 현재 main(#37 기본값)은 아직 재지 않았다. t2w-best 는 env 5개를 함께 켜 2,034 tok/s 였고, 분할 단독의 몫은 알려져 있지 않다.

## 2. 사전 대조 결과 (계획 단계에서 확인)

- **이식 가능.** `e47aa780bc`(T2W 베이스)는 우리 HEAD 의 조상이다. `git diff e47aa780bc HEAD -- sparse_attn_indexer.py` 는 SM12x `use_persistent_topk` hunk 하나(+7/−4, 694 행대)뿐이다. T2W 패치의 이 hunk 는 우리가 이미 가진 것이라 빼고, 나머지 hunk 4개(import, 헬퍼 블록, 루프 훅, `__init__` 검사)만 가져온다.
- 훅이 쓰는 이름은 모두 루프 스코프에 있다: `use_pcp`(318), `dcp_world_size`, `candidate_blocks`, `candidate_block_size`, `candidate_write`(327), `k_quant`/`k_scale`(473-474, 훅보다 앞에서 K gather 가 끝난다). `SparseAttnIndexer.__init__` 의 DeepGEMM 검사는 863 행.
- max10 초안 `tests/v1/attention/test_indexer_tp_split.py` 168줄(CPU 3개)을 읽었다. 헬퍼 이름·시그니처가 T2W 패치와 같아 그대로 쓸 수 있다. `tests/v1/attention/` 는 우리 트리에 있다.
- **재계산 자체가 비트 단위로 같지 않다.** 헤드 `casebench.py` 의 `run_logits` 주석에 "solo recomputes of a near-tie prompt already moved the top logprob by 0.3 nats" 가 남아 있다. 분할이 없어도 같은 프롬프트를 다시 계산하면 흔들린다는 뜻이다. 그래서 출력 동일성은 분할 꺼짐끼리의 대조군과 함께 봐야 한다(§5-2).
- all-gather 크기: 4096 행 청크(MNBT 4096) 하나당 층 7개 × 4096×512 int32(각 7.8 MiB)에 20 층 4096×(512+2048) int32 39 MiB 를 더해 약 94 MiB 가 모인다. 랭크당 1024 행이라 64 정렬 패딩은 없다. 128K 청크 하나에 2.8 s 가 걸리니, 이 통신은 점수 계산의 3/4 절감보다 작을 것으로 본다. 실측으로 판정한다.

## 3. 바꿀 것

### 3-1. 코드 (커밋 a: 기본 꺼짐, 행동 불변)

`sparse_attn_indexer.py` 에 T2W hunk 4개를 로직 그대로 옮긴다.

- (a) `import os`, `get_tp_group`, `model_parallel_is_initialized` import.
- (b) `_merge_dcp_topk_global` 뒤 헬퍼 블록: `_dsv41_check_tp_split_env_consistent`, `_dsv41_tp_split_world`, `_dsv41_tp_split_bounds`, `_dsv41_prefill_rows_local`, `_dsv41_scatter_gathered`, `_dsv41_prefill_chunk_tp_split`. env 는 import 시 읽는다(`DSV41_INDEXER_TP_SPLIT` 기본 0, `DSV41_INDEXER_TP_SPLIT_MIN` 기본 512). 이름은 T2W 와 같게 둔다.
- (c) 루프의 `topk_indices = …` 직후 훅: world > 1 이고 `local_total_seq_lens > 0` 이면 분할 경로를 탄 뒤 `continue` 한다. DCP 가 1 이라 `_merge_dcp_topk_global` 은 어차피 no-op 이다.
- (d) `__init__` DeepGEMM 검사 뒤 `_dsv41_check_tp_split_env_consistent()`.

T2W 원본의 긴 배너 주석은 이 저장소 관례(AGENTS.md "주석 최소화")에 맞춰 출처 한 줄과 핵심 불변 조건 두 줄로 줄인다. 함수 docstring 은 둔다. 88자 초과 주석 줄(훅 위 한 줄)은 줄인다. 로직은 한 글자도 바꾸지 않는다. T2W 트리와 비교 가능해야 해서다.

분할 경로가 지키는 불변 조건(코드 읽기로 확인):

- 행 슬라이스만 다르고, 커널 인자(`ks/ke` 는 청크 K 기준 절대 오프셋)와 −1 선채움은 같다.
- 모든 랭크가 같은 게이트 입력(env 는 기동 때 비교, 행 수는 방송된 스케줄러 출력)으로 같은 분기와 같은 순서의 collective 를 탄다.
- 캡처 중이면 assert 로 막는다. 디코드 행과 행 수 < 512 청크(혼합 배치의 짧은 프리필)는 분할하지 않는다.
- XPU/ROCm(`is_cuda()` 거짓), DCP/PCP 는 분할하지 않는다.

### 3-2. 테스트 (커밋 a)

`tests/v1/attention/test_indexer_tp_split.py`: max10 초안 3개를 그대로 가져오고, 기동 거부 경로를 보는 테스트를 하나 더한다.

- `test_env_mismatch_refuses_start`: `model_parallel_is_initialized`, `get_tp_group`(world 4, `cpu_group` 더미), `torch.distributed.all_gather_object`(한 랭크만 다른 튜플을 채우는 가짜)를 monkeypatch 한다. `RuntimeError` 가 나는지, 전부 같으면 통과하는지, 검사 플래그가 한 번만 도는지 본다. 이슈 완료 기준인 "불일치 시 기동 거부"를 단위 수준에서 고정한다.

실행 장소: 파일이 `vllm._custom_ops`, DeepGEMM, triton 을 import 해서 macOS 로컬(워크트리에 .venv 없음)에서는 못 돌린다. **헤드에서 서버를 내린 창에서만** `~/vllm-dsv41-venv/bin/python -m pytest tests/v1/attention/test_indexer_tp_split.py -q` 로 돌린다(CPU, 수 초). 로컬에서는 `pre-commit run --files <변경 파일>`(ruff/mypy) 만 돌린다.

### 3-3. 배선 (커밋 a)

- `deploy/gb10-cluster/dsv41/dsv41.env`: `DSV41_INDEXER_TP_SPLIT=${DSV41_INDEXER_TP_SPLIT:-0}` 한 줄과 주석을 넣는다. `serve-node.sh:11` 이 `set -a; source dsv41.env` 로 모두 export 하므로 `serve-node.sh` 는 고치지 않는다. 같은 방식인 `DSPARK_DRAFT_PRUNE` 가 선례다. 매니저 §5 의 "serve-node.sh 에서 export" 는 이 경로로 이미 충족된다.
- `dsv41_ctl.sh:15` KNOBS 에 `DSV41_INDEXER_TP_SPLIT` 을 추가한다. `_MIN` 은 기본 512 고정이라 env/KNOBS 에 넣지 않는다(매니저 §6-3).
- 확인: `bash -n` 두 스크립트, `selftest_caps.sh`.

### 3-4. 채택 시 (커밋 b)

`dsv41.env` 기본값을 1 로 바꾸고 주석에 수치를 적는다. `README.md` 75 행 근처 #37 문단 뒤에 한 문단을 넣는다. 결과는 `.notes/2026-09-15-issue-38-indexer-tp-split/results.md` 에 둔다. 미채택이면 (a) 만 기본 꺼짐으로 머지한다. #15 의 스케줄러 캡과 같은 선례이고, T2W 비교와 이후 재측정 레버로 남긴다.

## 4. 영향 사이트

- 호출자: `SparseAttnIndexer.forward_cuda` → `torch.ops.vllm.sparse_attn_indexer` → `sparse_attn_indexer()`. 쓰는 곳은 V4.1 `vllm/models/deepseek_v4_1/attention.py:1122`(인덱서 원천 층 8개, 20 층이 `candidate_write`)이고, 같은 op 를 V3.2/V4/GLM5next/HY-V4/deepseek_v2 도 쓴다. 이들은 env 기본 0 이면 게이트가 `world=1` 을 돌려주고 경로가 같다. env 를 켜도 분할은 CUDA·TP>1·DCP/PCP 없음에서만 켜진다.
- `__init__` 검사는 env 가 꺼져 있어도 TP>1 이면 기동 때 `all_gather_object` 를 **한 번** 한다(TP CPU 그룹, 튜플 하나). 모든 랭크가 모델 빌드 중에 같은 지점을 지나므로 교착은 없다. 비용은 무시할 만하다.
- 소비자: `topk_indices_buffer`(sparse MLA 백엔드), `candidate_blocks`(21 층 이후 인덱서). 채우는 행과 열 범위는 비분할과 같다.
- 드래프트: `num_nextn_predict_layers` 3. 드래프트 프리필이 같은 op 를 타면 거기서도 분할이 걸리지만 불변 조건은 같다. 수락 길이 측정(§5-2)이 이 경로까지 덮는다.
- 노드 반영: 파이썬·셸·env 만 바뀐다. 노드 4대에 `git format-patch | git am`(워커 3대 `-c user.name -c user.email`, README hunk 는 `--exclude`). `.so` 재빌드는 없다.

## 5. 검증 방법

### 5-1. 로컬

`pre-commit run --files vllm/model_executor/layers/sparse_attn_indexer.py tests/v1/attention/test_indexer_tp_split.py`, `bash -n deploy/gb10-cluster/dsv41/{dsv41_ctl.sh,serve-node.sh}`, `selftest_caps.sh`.

### 5-2. 출력 동일성 프로브 (새 일회성 스크립트, stdlib)

`.notes/2026-09-15-issue-38-indexer-tp-split/greedy_probe.py.txt` 에 두고(#37 의 `.py.txt` 선례), 헤드에서 `python3` 으로 돌린다.

- 고정 시드로 만든 8K, 32K 프롬프트 두 개를 쓴다. 코드/레코드 본문에 긴 답이 필요한 지시를 붙인다.
- 요청: `temperature 0`, `max_tokens 64`, `thinking False`, `logprobs true`, `top_logprobs 5`. 매 요청마다 새 `cache_salt` 를 줘서 prefix 캐시·SSD 티어 복원을 피한다. 프롬프트 토큰은 그대로이고 해시만 달라진다.
- 부팅마다 프롬프트당 2회. 같은 부팅의 2회 비교가 **대조군**이다.
- 기록: 텍스트, 토큰 id 열, 첫 불일치 위치, 토큰별 top-1 logprob 차 최대값.
- 판정:
    - 꺼짐끼리 같고 켜짐이 꺼짐과 같음 → 동일.
    - 꺼짐끼리 같은데 켜짐만 다름 → **미채택**, 원인 기록(매니저 §2).
    - 꺼짐끼리도 다름 → 켜짐의 차이가 대조군 범위(첫 불일치 위치, logprob 차) 안인지 비교해 기록하고, §7-1 에 따라 오너 판단으로 올린다.

### 5-3. 클러스터 A/B (TP=4, 한 번에 한 부팅, 단계마다 전경 실행 후 확인, 백그라운드 체인 없음)

공통 절차:

1. `dsv41_ctl.sh headroom`(헤드 ≥ 5.2).
2. 외부 클라이언트가 없는지 확인(r0.log 최근 `POST /v1` 출처).
3. `stop`.
4. 서버가 내려간 창에서만 헤드 pytest.
5. `[오버라이드] dsv41_ctl.sh start`.
6. r0.log 에서 부팅 인자와 분할 로그(`DSV41_INDEXER_TP_SPLIT on: … 4 TP ranks`) 유무를 확인.
7. health 200, 웜업 1회.

| 순서 | 구성 | 잰다 |
| --- | --- | --- |
| 0 | 노드 4대 `git am`(a) → 헤드 pytest | 테스트 4개 통과 |
| M `i38-mismatch` | 워커 gx10-27c4 한 대만 `systemctl --user set-environment DSV41_INDEXER_TP_SPLIT=1` 뒤 기본값 `start` | 4랭크 로그에 `RuntimeError: DSV41_INDEXER_TP_SPLIT … differ across TP ranks` 가 모델 빌드 단계(가중치 로드 전)에서 뜨는지 확인 → `stop` → `systemctl --user unset-environment DSV41_INDEXER_TP_SPLIT` → `show-environment` 에서 사라졌는지 확인. 파일 수정은 없다. 수 분 안에 끝나는 짧은 부팅이다. |
| A `i38-off` | 오버라이드 없음(분할 꺼짐, 현재 main 기본값) | casebench solo S32 ×2, S128 ×2(`run_suite.sh` 인자 그대로), decodebench prose,code c1 gen 256, greedy 프로브(§5-2), S128 중 헤드 MemAvailable 최저 |
| B `i38-on` | `DSV41_INDEXER_TP_SPLIT=1`(ctl KNOBS) | S8, S32 ×2, S128 ×2, decodebench prose,code c1 gen 256, greedy 프로브, `longctx.py --records 10700`(158K 니들 + 후속 턴 TTFT), 헤드 MemAvailable 최저 |
| 최종 | 채택이면 (b) 를 노드에 `git am` 뒤 오버라이드 없이 `start`, 미채택이면 기본값(0) 그대로 `start` | health 200, 분할 로그 유무(채택=on), decodebench c1 짧게 확인, `caps` 4대 active 1989 MHz(부하 중), 헤드 MemAvailable 기록 — 골든룰 |

부팅은 전체 3회(A, B, 최종)에 짧은 거부 부팅 M 1회를 더한다. 매니저 예산 ≤ 4 안이다.

### 5-4. 채택 판정 (모두 충족)

1. 출력 동일성 §5-2 통과.
2. 수락 길이: 산문 c1 `tokens_per_chunk` 2.91~3.05, 코드 c1 5.82 유지(`decode.jsonl`).
3. 158K 니들 정답.
4. S128(그리고 방향 확인용 S32)에서 켜짐 − 꺼짐 이득이 두 반복 모두 잡음(약 3%) 밖. 1,700 tok/s 도달 여부는 기록만 하고 채택 조건으로 삼지 않는다(매니저 §1).
5. S128 중 헤드 MemAvailable 최저 ≥ 3.0 GiB. 분할은 랭크당 logits 순간 텐서를 1/4 로 줄이므로 나빠지지 않을 것으로 본다.
6. 기동 거부 확인(M) 통과. 이것은 채택 여부와 무관하게 (a) 머지 조건이다.

## 6. 이번에 하지 않을 것

- `DSV41_ENGRAM_FAST`, 읽기 스레드 128, `NCCL_MAX_NCHANNELS=8`, `expandable_segments:False` 등 t2w-best 의 다른 env 는 가져오지 않는다(이슈 본문). 따라서 2,034 tok/s 재현을 기대하지 않는다.
- #40 범위(`vllm/v1/attention/backend.py`, `vllm/v1/attention/ops/metadata.py` token_to_req_indices Triton 커널)는 max10 초안에 같이 있지만 섞지 않는다.
- `DSV41_INDEXER_TP_SPLIT_MIN` 스윕은 하지 않는다(512 고정).
- 비분할 루프 본문을 `_dsv41_prefill_rows_local` 로 합치는 중복 제거는 하지 않는다. T2W 트리와 줄 단위로 비교할 수 있게 원형을 유지한다.
- 디코드 경로, DCP/PCP, XPU 경로는 바꾸지 않는다.

## 7. plan-approve 에서 정할 것

1. **대조군(꺼짐끼리)부터 비트 단위로 다를 때의 채택 기준.** 근거: 헤드 casebench 주석에 near-tie 프롬프트를 다시 계산하면 top logprob 가 0.3 nats 흔들린다고 남아 있다. 이 경우 "켜짐 = 꺼짐 텍스트 동일"은 분할과 무관하게 실패할 수 있다.
   - 권고: 켜짐과 꺼짐의 첫 불일치 위치·logprob 차가 대조군 범위 안이고 수락 길이·니들이 정상이면 채택 가능으로 본다.
   - 켜짐만 대조군보다 일찍 또는 크게 갈라지면 미채택.
   - 오너가 비트 동일을 요구하면, 대조군이 흔들리는 경우 판정을 보류하고 결과만 기록한다.
