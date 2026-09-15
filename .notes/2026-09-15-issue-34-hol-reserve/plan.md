# 이슈 #34 계획 — 긴 프리필 중 짧은 요청의 HOL 을 "필요한 만큼만 예약"(`DSPARK_SHORT_RESERVE`)으로 줄이기

작성 2026-09-15 KST, 워크트리 `issue-34-hol-reserve`(base `main` 9fd8bcc1d3). 매니저 코멘트 icmt-35c04236 의 정책·레시피·제약을 따른다. 코드는 아직 건드리지 않았다. 이 단계의 조사는 읽기 전용이다: 로컬 `scheduler.py` / `test_scheduler.py` / `deploy/gb10-cluster/dsv41/*`, 그리고 헤드의 `~/dsv41-prep/mixlong.sh` 한 번 읽기.

## 1. 현재 동작과 문제

128K 콜드 프리필이 도는 동안 들어온 16토큰 요청이 약 65~70 s 를 기다린다. 대기 시간이 프리필 전체 시간과 같다. 이유는 `vllm/v1/core/sched/scheduler.py` 의 두 군데다.

- **입장이 막힌다.** 실행 루프(`:616`)가 `self.running` 순서대로 돌며 각 요청에 `min(남은 토큰, token_budget, …)`(`:656`)를 준다. 프로덕션은 `LPTT` 가 비어 있어 `self._long_prefill_threshold` 가 0 이고(`_effective_long_prefill_threshold`, `:538`), `LPTT_MIXED`/`PPCAP`/`DECODE_STEPS` 도 전부 꺼져 있다. 그래서 128K 프리필 하나가 MNBT 4096 을 매 스텝 전부 가져간다. 대기 루프(`:842`)는 `token_budget > 0` 일 때만 도니, 짧은 요청은 프리필의 마지막 청크가 예산을 남길 때까지 입장하지 못한다.
- **입장해도 안 끝난다.** 새 요청은 `self.running` 끝에 붙는다(`:1232`). 다음 스텝부터 앞에 선 프리필이 다시 4096 을 다 쓰면 실행 루프가 `token_budget > 0` 조건에서 끝나고 뒤의 디코드는 토큰을 못 받는다. casebench `hol` 의 `short_wall_s` 는 완료까지 재므로(`casebench.py:642`, `max_tokens 16`, thinking off), 입장만 시키는 정책으로는 숫자가 안 내려간다.

기존 노브가 이 문제에 안 맞는 이유는 셋 다 **고정량**을 깎기 때문이다. `LPTT` 는 단독 프리필도 깎고, `LPTT_MIXED`(`:538-552`)는 디코드가 하나라도 돌면 고정 상한을 걸어 DS4F 2048 에서 프리필 −33% 였고, `DECODE_STEPS`(`:611`, `:643`)는 디코드 전용 스텝을 끼워 S4 에서 878 tok/s(총시간 +10%)가 됐다. 오너 기준이 총시간이라 전부 미채택으로 남아 있다.

## 2. 계획 단계에서 확인한 것

- **`request.is_prefill_chunk` 를 판별자로 쓸 수 있다.** `_update_after_schedule`(`:1544`)에서 스텝마다 `num_computed_tokens < num_tokens + num_output_placeholders` 로 갱신된다. `schedule()` 진입 시점에는 직전 스텝 기준값이고, 이미 `:611`·`:643` 이 같은 뜻("지금 프리필 중")으로 쓴다. 새 판별자를 만들지 않고 이 관례를 따른다.
- **대기 큐는 앞에서부터 훑을 수 있다.** `FCFSRequestQueue`(deque)와 `PriorityRequestQueue` 둘 다 `__iter__`(`request_queue.py:126`, `:194`)를 우선순위 순으로 제공한다. 훑기만 하면 부작용이 없다.
- **대기 요청의 프리픽스 적중은 입장 전에 모른다.** 대기 중 `num_computed_tokens` 는 0 이고, 적중은 `_get_local_prefix_cache_hit`(`:505`) → `kv_cache_manager.get_computed_blocks` 를 대기 루프 안에서 부를 때 비로소 나온다. 그래서 1차는 프롬프트 길이로만 판정한다. **이것이 기대치에 직접 영향을 준다**: AG4 의 새 턴은 신규 계산이 2~8K 여도 프롬프트 자체는 12K 에서 시작해 자라므로 `SHORT_RESERVE=4096` 에서 예약 대상이 아니다. 즉 MIXLONG 은 좋아지지 않고 **기준선과 같게 나오는 것이 정상**이다(판정 기준은 "+2% 안"이지 "개선"이 아니다). HOL 의 16토큰 요청만 예약 대상이다.
- **`assert token_budget >= 0`(`:1299`) 은 자동으로 지켜진다.** 이 정책은 프리필 청크의 상한을 *더 낮추기만* 하므로 예산을 초과시킬 수 없다.
- **MoE 효율은 안 바뀐다.** 스텝 총 토큰은 여전히 MNBT 이하이고, 예약분은 다른 요청이 같은 스텝에서 쓴다. 32K 프리필 2개/4개 동시가 1,505/1,463 tok/s 로 단독과 같은 수준이라는 `.notes/2026-09-13-dsv41-mixed-prefill/results.md` 가 근거다.
- **`mixlong.sh` 는 `--out` 을 넘기지 않는다.** 헤드 `~/dsv41-prep/mixlong.sh` 를 읽어 확인했다. 두 레코드가 기본 경로 `~/dsv41-prep/bench/casebench.jsonl` 로 간다. 매니저가 지정한 `~/sglang-cmp/results/casebench.jsonl` 에 모으려면 경로를 넘기는 사본이 필요하다(§5-1).
- **AG4 워크로드는 `mixlong.sh` 의 AG 라인과 `agentbench-run.sh` 의 AG4 라인이 동일하다**(lanes 4, turns 6, stagger 8, 12000 / 2000~8000 / 150~900, `--work-seed w1`, thinking high, temperature default). 그래서 `casebench.py --mode agent` 를 직접 같은 인자로 부르면 #36~#38 의 208.0 s 와 비교 가능하다.
- **`~/dsv41-prep/i34/` 는 아직 없다.** 새로 만든다(기존 디렉터리를 비우지 않는다).

## 3. 바꿀 것

### 3-1. 코드 (커밋 a: 기본 꺼짐, 행동 불변)

`vllm/v1/core/sched/scheduler.py` 한 파일.

**(a) env 상수** — `:95` 의 `DSPARK_DECODE_STEPS_PER_PREFILL` 아래, 같은 `_env_int` 관례로 둘을 더한다.

```python
# Need-based reservation (issue #34): while a long prefill runs, hold back only
# as many tokens as the decodes queued behind it and the short waiting requests
# need this step, so a short request is not stuck for the whole prefill.
# 0 disables. The value is also the "short" cutoff on a waiting request's new
# tokens.
DSPARK_SHORT_RESERVE = _env_int("DSPARK_SHORT_RESERVE")
# How far into the waiting queue the reservation looks.
DSPARK_SHORT_RESERVE_SCAN = _env_int("DSPARK_SHORT_RESERVE_SCAN") or 8
```

**(b) 예약량 계산** — `_long_prefills_at_cap`(`:554`) 뒤에 메서드 하나.

```python
def _short_reserve_tokens(self, token_budget: int) -> int:
    """Tokens to hold back from prefill chunks for requests behind them."""
    if DSPARK_SHORT_RESERVE <= 0:
        return 0
    reserve = 0
    behind_prefill = False
    for request in self.running:
        if request.is_prefill_chunk:
            behind_prefill = True
        elif behind_prefill:
            reserve += (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
    if behind_prefill:
        for n, request in enumerate(self.waiting):
            if n >= DSPARK_SHORT_RESERVE_SCAN:
                break
            if self._is_blocked_waiting_status(request.status):
                continue
            num_new = request.num_tokens - request.num_computed_tokens
            if num_new <= DSPARK_SHORT_RESERVE:
                reserve += num_new
    return min(max(reserve, 0), token_budget // 2)
```

설계 근거:

- **프리필 앞에 선 디코드는 세지 않는다.** 실행 루프가 순서대로 도니 이미 예산을 먼저 받는다. 세면 이중 예약이다.
- **프리필 청크가 하나도 없으면 예약 0 이다.** 실행 루프가 예산을 다 쓸 일이 없으니 짧은 요청은 원래 바로 입장한다. 단독 프리필(S128)도 대기 요청이 없으면 0 이라 판정 기준 "S128 불변"이 코드 수준에서 보장된다.
- **차단 상태 대기 요청**(`WAITING_FOR_REMOTE_KVS` 등, `_is_blocked_waiting_status` `:2305`)은 이번 스텝에 입장하지 못하므로 제외한다. 예약했다가 안 쓰면 순손실이다.
- **예산 절반 캡**은 매니저 정책 그대로다. 실행 루프 첫 요청이 프리필일 때 상한이 0 이 되어 `num_new_tokens == 0` → `continue`(`:699`) 로 프리필이 통째로 굶는 것을 막는다.

**(c) `schedule()` 에서 한 번 계산** — `self._long_prefill_threshold = …`(`:584`) 옆에 `reserve_remaining = self._short_reserve_tokens(token_budget)` 를 둔다. 지역 변수로 두어 스케줄러 상태를 늘리지 않는다.

**(d) 실행 루프에서 프리필 청크만 자른다** — `:656` 의 `min(...)` 을 프리필 청크에 한해 `token_budget` 대신 `token_budget - reserve_remaining` 으로 바꾼다. `_long_prefill_threshold` 적용 뒤, `min(token_budget, input_budget - draft_slots)` 자리에서 한다. 스케줄 확정 직후(`:791` `token_budget -= num_new_tokens` 옆) 예약 소비자가 받아간 만큼 `reserve_remaining` 을 줄인다(음수 방지). 그래야 루프 뒤쪽의 두 번째 프리필이 이미 쓰인 예약분만큼 과도하게 깎이지 않는다.

대기 루프(`:842` 이후)는 **손대지 않는다.** 예약은 "예산을 남겨두는" 것이고, 남은 예산은 기존 `request_token_budget`(`:1071`) 경로가 그대로 쓴다.

### 3-2. 지키는 불변 조건 (코드 읽기로 확인, 테스트로 고정)

- `assert token_budget >= 0`(`:1299`), `assert input_budget >= 0`, `total_num_scheduled_tokens <= max_num_scheduled_tokens`: 상한을 낮추기만 하므로 자동.
- FCFS 순서: `self.running` 순회 순서와 대기 큐 pop 순서를 바꾸지 않는다. 기존 루프가 이미 `num_new_tokens == 0` 에서 `break` 가 아니라 `continue` 로 넘어가는 것과 같은 성격의 완화다.
- 선점(`:769` `self.running.pop()`): 순서를 안 바꾸므로 선점 피해자 선택이 달라지지 않는다. 선점 시 복원되는 `token_budget += restored`(`:753`)와 `reserve_remaining` 은 독립이다(예약은 상한일 뿐 잔액이 아님).
- 멀티모듈 MTP 프리필 룩어헤드(`_reserve_prefill_lookahead`, `:517`): 예약 상한을 먼저 적용하고 기존 순서대로 룩어헤드 보정이 뒤에 온다. 청크가 작아져 프리필 끝 근처에 걸리는 경우는 기존 `token_budget` 만으로도 생기던 상황과 같은 부류이고, 128K 프리필의 중간 청크는 끝에서 멀어 해당 없다.
- 꺼짐(`DSPARK_SHORT_RESERVE=0`)이면 `_short_reserve_tokens` 가 즉시 0 을 돌려주고 `min()` 인자가 `token_budget` 그대로라 `num_scheduled_tokens` 가 비트 단위로 동일하다.

### 3-3. 테스트 (커밋 a)

`tests/v1/core/test_scheduler.py`, 기존 `test_lptt_mixed_caps_only_competing_prefills`(`:6345`) 옆에 붙인다. 파일의 `create_scheduler` / `create_requests` / `monkeypatch.setattr(sched_module, …)` 관례를 그대로 쓴다. 최소 4개:

1. `test_short_reserve_admits_waiting_short_request` — 대기에 짧은 요청이 있을 때 프리필 청크가 예약분만큼 줄고 짧은 요청이 **같은 스텝에** 입장하는가.
2. `test_short_reserve_feeds_decode_behind_prefill` — 프리필 뒤에 선 디코드가 다음 스텝에 토큰을 받는가(입장 스텝 → `update_from_output` → 다음 스텝에서 디코드에 `num_scheduled_tokens` 가 잡히는지). `test_decode_steps_per_prefill_defers_prefill_chunks`(`:6382`)의 `step()` 헬퍼 패턴을 재사용한다.
3. `test_short_reserve_off_matches_baseline` — 0 이면 `num_scheduled_tokens` 가 노브 없는 스케줄러와 같은가(같은 시나리오 두 스케줄러 대조).
4. `test_short_reserve_ignores_long_waiting_request` — 대기 요청의 신규 토큰이 임계 초과면 예약하지 않는가(= 단독 긴 프리필의 청크가 안 줄어듦, S128 불변의 단위 수준 고정).

실행: **서버가 내려간 헤드**에서 `~/vllm-dsv41-venv/bin/python -m pytest tests/v1/core/test_scheduler.py -k "short_reserve or lptt or ppcap or decode_steps" -q`. 로컬 macOS 워크트리에는 venv 가 없고 이 파일은 CUDA 계열 import 를 끌어온다. 로컬에서는 `pre-commit run --files <변경 파일>`(ruff/mypy) 만 돌린다.

### 3-4. 배선 (커밋 a)

- `deploy/gb10-cluster/dsv41/dsv41.env`: `DECODE_STEPS` 줄 아래에 `SHORT_RESERVE=${SHORT_RESERVE:-}` (빈 값 = 꺼짐) 와 주석.
- `deploy/gb10-cluster/dsv41/serve-node.sh`: `:85` 의 `DECODE_STEPS` export 옆에 `[ -n "${SHORT_RESERVE:-}" ] && export DSPARK_SHORT_RESERVE="$SHORT_RESERVE"`. (`LPTT_MIXED`/`PPCAP` 과 같은 이름 매핑 방식. `DSV41_*` 처럼 이름이 같은 노브만 `set -a` 로 자동 export 되므로 여기서는 명시 export 가 맞다.)
- `deploy/gb10-cluster/dsv41/dsv41_ctl.sh:15` KNOBS 목록에 `SHORT_RESERVE` 추가. `SCAN` 은 8 고정이라 넣지 않는다(#38 의 `_MIN` 과 같은 선례).
- 확인: 두 스크립트 `bash -n`, `selftest_caps.sh`.

### 3-5. 채택 시 (커밋 b)

`dsv41.env` 기본값을 `SHORT_RESERVE=${SHORT_RESERVE:-4096}` 으로 바꾸고 주석에 실측치를 적는다. `README.md:93` 의 "The scheduler caps … stay unset by default" 문단을 고쳐 `SHORT_RESERVE` 만 예외로 켜져 있다고 쓰고, 왜 고정 상한 노브들과 다른지(필요한 만큼만, 스텝 총 토큰 불변) 두세 줄로 적는다. 결과는 `.notes/2026-09-15-issue-34-hol-reserve/results.md`.

미채택이면 커밋 a 만 기본 꺼짐으로 머지한다. `LPTT_MIXED`/`PPCAP`/`DECODE_STEPS` 와 같은 선례이고, 이후 재측정 레버로 남긴다. 어느 쪽이든 결과는 results.md 에 쓴다.

## 4. 영향 사이트

- **호출자**: `Scheduler.schedule()` 하나. `EngineCore.step()`/`step_with_batch_queue()` 가 부른다. 시그니처·반환형은 안 바뀐다.
- **소비자**: `SchedulerOutput.num_scheduled_tokens` → 모델 러너(V2 러너, `gpu_model_runner`), 스펙 디코드 입력, `_update_after_schedule`. 값의 *분포*만 달라지고 계약(합 ≤ MNBT, 요청당 > 0)은 유지된다.
- **다른 스케줄러 구현**: `vllm/v1/core/sched/` 의 다른 정책 클래스는 이 루프를 상속하지 않는다(별도 `schedule()`). 영향 없음.
- **다른 배포**: 이 저장소의 다른 런처/테스트는 `DSPARK_SHORT_RESERVE` 를 설정하지 않으므로 상수 0 → 경로 동일.
- **건드리지 않는 것**: 샘플러, 모델, 어텐션, KV 매니저, 커넥터.

## 5. 검증

한 단계씩 포그라운드로 돌린다. 백그라운드 체인 없음. 외부 클라이언트가 0 인지 먼저 확인하고 시작한다(지표는 전역이라 오염된다).

### 5-1. 기준선 재측정 (부팅 없음, 지금 프로덕션 :8888)

헤드에 `~/dsv41-prep/i34/` 를 만들고, 거기에 `--out ~/sglang-cmp/results/casebench.jsonl` 를 넘기는 `mixlong_i34.sh` 사본을 둔다(원본 `~/dsv41-prep/mixlong.sh` 는 안 건드린다). 태그 접두사 `i34-base`.

1. 웜업: `--mode solo --prefill-tokens 16000`, `--mode mixed --prefill-tokens 6000 --decodes 2`
2. `--mode hol --prefill-tokens 128000` **2회** (`i34-base-HOL1/2`)
3. `mixlong_i34.sh i34-base <label> 8888` **1회** (약 6분, `MX4`+`MXL` 두 레코드)
4. AG4 1회: `--mode agent --lanes 4 --turns 6 --stagger 8 --start-tokens 12000 --turn-tokens-min 2000 --turn-tokens-max 8000 --gen-min 150 --gen-max 900 --work-seed w1 --thinking high --temperature default`
5. **LOGS 1회**(`--mode logits --prefills 3 --prefill-tokens 16000 --decodes 2`)와 **S128 1회**(`--mode solo --prefill-tokens 128000`)도 함께 잰다. 매니저 §4 의 기준선 목록에는 없지만 판정 기준에 "S128 불변"과 "로짓 뒤집힘 0"이 있어 대조군이 없으면 판정할 수 없다. 둘 다 합쳐 2분 남짓이라 추가한다.

### 5-2. 실험 부팅 (부팅 1회차)

`dsv41_ctl.sh stop` → 4 노드에 `git format-patch` → `git am` 배포(README 제외, 워커 3대는 `-c user.name/-c user.email`) → **서버 정지 상태의 헤드**에서 §3-3 의 pytest → `SHORT_RESERVE=4096 dsv41_ctl.sh start` → 웜업(32K solo, 8K mixed) → HOL 2회 / mixlong 1회 / AG4 1회 / LOGS 1회 / S128 1회. 태그 접두사 `i34-sr4096`.

### 5-3. 판정

| 항목 | 기준 |
| --- | --- |
| HOL `short_wall_s` | ≤ 15 s (2회 모두) |
| MIXLONG 총시간 | 재측정 기준선 +2% 이내 |
| AG4 단독 총시간 | ±3% |
| S128 단독 프리필 tok/s | 불변(노이즈 범위) |
| 로짓 프로브 | 상위 토큰 뒤집힘 0 |
| 헤드 MemAvailable 최저 | ≥ 3.0 GiB |

§2 에서 적었듯 MIXLONG 은 개선이 아니라 **동률이 기대치**다. HOL 예상은 입장 스텝(약 2.3 s) + 디코드 몇 스텝(프리필 청크에 얹혀 스텝당 약 2.2 s) = 5~10 s.

### 5-4. 마무리 (부팅 2회차)

통과면 `dsv41.env` 기본값 4096 으로 커밋하고 **오버라이드 없이** 재기동, 미채택이면 기본값 그대로 재기동. 부팅 예산 3회 중 2회를 쓰고 1회를 예비로 남긴다.

골든룰(둘 중 어느 쪽이든 마지막에):

- `:8888` health 200
- `dsv41_ctl.sh caps` 4대 active / 1989 MHz
- `systemctl --user show dsv41-serve -p Environment` 비어 있음
- 4 노드 트리 dirty 0

금지 준수: `nvidia-smi -pm/-lgc/-rgc` 안 씀, `gpu-clock-cap.service` 안 멈춤, 라이브 서버 노드에서 pytest/torch 안 돌림, 글롭 삭제 안 함, 백그라운드 체인 안 씀.

## 6. 이번에 하지 않을 것

- **프리픽스 적중을 미리 조회하는 2차안**(대기 요청 앞쪽에만 `get_computed_blocks` 를 선조회해 신규 계산량으로 판정). 매니저가 "MIXLONG 이 안 좋아지면" 조건으로 달아둔 것인데, §2 대로 1차안에서 MIXLONG 은 애초에 좋아지지 않는 것이 정상이고 판정 기준도 "개선"이 아니다. 이걸 넣으면 KV 매니저 부작용 확인 + 부팅 1회가 더 필요하다. §7 에서 오너 판단을 받는다.
- **실행 루프 2패스 대안**(프리필 끝난 요청을 먼저 채우고 프리필 청크를 나중에). 매니저가 제시한 두 안 중 예약 방식을 고른 이유: 2패스는 (a) 대기 중인 짧은 요청에는 아무 효과가 없어 입장 문제를 못 푼다, (b) 선점이 `self.running.pop()` 로 뒤에서부터 집어가는데 2패스가 순서를 흔들면 방금 스케줄한 디코드가 피해자가 될 수 있다, (c) `req_index` 보정 로직(`:744-747`)을 두 배로 만든다. 예약 방식은 상한만 낮춰 세 문제를 모두 피한다.
- 디코드 분배(`DECODE_STEPS`) 켜기. 이슈 본문이 명시적으로 배제한다.
- 새 벤치 도구 추가. `casebench.py` 기존 모드만 쓴다.

## 7. 오너 판단이 필요한 것

**1차안이 HOL 은 통과하고 MIXLONG 은 기준선과 동률로 나올 때, 2차안(프리픽스 적중 선조회)을 이번 이슈에서 이어갈 것인가?**

- 근거: AG4/MIXLONG 의 새 턴은 프롬프트가 12K 이상이라 프롬프트 길이 판정으로는 예약 대상이 아니다(§2). 그 턴들의 실제 신규 계산은 2~8K 이므로, 선조회를 넣으면 "긴 세션 재개 중 다른 에이전트의 새 턴"도 짧은 요청으로 인정되어 MIXLONG 이 실제로 줄어들 여지가 있다.
- 비용: `get_computed_blocks` 를 대기 요청에 미리 부르는 것이 통계 외 부작용이 없는지 먼저 확인해야 하고(블록 터치/캐시 히트 카운터), 부팅이 1회 더 든다(예비 1회를 소진).
- **권고: 하지 않는다.** 이번 이슈의 완료 조건은 HOL ≤ 15 s + MIXLONG 무회귀다. 2차안은 목표가 다르고(총시간 개선) 위험 표면이 넓다. 별도 이슈로 넘기고, 이번에는 1차안 결과와 함께 "다음 레버"로 results.md 에 기록하는 쪽을 제안한다.
