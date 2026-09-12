# 이슈 #27 계획 — 승격 대기(HIT_PENDING) 요청의 매 스텝 재조회를 캐시 상태 에포크로 건너뛰기

기준: main caadeffc95, 워크트리 `issue-27-pending-lookup`. 매니저 조사 icmt-bf16290f 의 레시피를 따르되 아래 두 곳을 고쳤다(§2-1 카운터 위치, §2-4 RETRY 플래그 전달 방식).

## 0. 순서 — 측정이 먼저

1. **Step 0 실측(코드 전)**: 지금 떠 있는 채택 구성 :8889 에서 대조(복원 없는 디코드) → 493K 콜드 복원 1건을 얹은 디코드 → 재기동 후 2세션 동시 복원(P13+S2)을 얹은 디코드. 계측은 §4-1.
2. **판정**: §4-1 의 "노이즈 안" 규칙에 들면 코드 없이 결과 기록 + README 한 단락으로 마감(매니저가 정당한 결말로 인정). 아니면 3으로.
3. 구현(§2) → 단위 테스트(서버 정지 창) → 노드 `git am` → 같은 측정 반복 → #25 게이트 → 채택 구성 복구.

## 1. 현재 동작과 문제

- 코어 스케줄러는 커넥터가 `None` 을 돌려준 요청을 `step_skipped_waiting` 에 되돌리고(`vllm/v1/core/sched/scheduler.py:865-875`) 다음 스텝에 같은 요청을 처음부터 다시 묻는다.
- `get_num_new_matched_tokens`(`offloading/scheduler.py:1024-1078`)는 `transfer_jobs` 가 비어 있으면 매 스텝 `_lookup` → `_lookup_complete_chunks` → `_maximal_prefix_lookup`(`:636-668`)/`_sliding_window_lookup`(`:670-701`)을 돈다. `HIT_PENDING` 은 `defer_lookup=True` 만 표시하고 break 하지 않으므로 full-attention 키 전부(493K 세션 ≈ 11.6K)를 끝까지 훑고, 이어 `_touch`(`:703-735`)가 같은 키를 LRU `move_to_end` 한다.
- 승격 완료는 `TieringOffloadingManager._process_finished_jobs` → `_complete_promotion`(`tiering/manager.py:277-345`) → `primary_tier.complete_write`(= `CPUOffloadingManager.complete_store`, `tiering/manager.py:128` 별칭) 로 블록이 ready 가 되고, 다음 스텝의 재조회에서 HIT 로 바뀐다. 요청에 알리는 신호는 없고 폴링뿐이다.
- #25 뒤 승격 job 은 0.32 s 라 이 비용이 얹히는 창은 요청당 RETRY 1스텝 + 약 0.32 s 다. 매니저 추정은 11.6K 키 한 패스 25~60 ms(순수 파이썬, EngineCore 메인 스레드)로, 디코드 스텝(약 21 ms)의 2~4배가 복원 1건당 0.3~0.6 s 동안 매 스텝 얹힌다. **실측은 없다** → Step 0.

### 불변 조건(건드리지 않을 것)

- `_touch` 는 대기 중에도 매 스텝 유지. 이슈 본문의 "touch 생략해도 LRU 불변" 은 CPU 풀이 빡빡한 2세션 동시 복원에서 틀리다(매니저 3절: A 의 HIT 블록이 B 의 `prepare_write` 퇴거에 먼저 밀림). touch 는 패스 비용의 1/10 이하.
- `RETRY`(fs 비동기 조회가 다음 스텝에 결과를 냄)는 매 스텝 재조회해야 한다. 단축은 `HIT_PENDING` 만으로 `None` 이 된 경우에만.
- `update_num_hit_chunks`, `LOOKUP_ASYNC_DELAY`(첫 지연 시각→해소) 의미 불변. `LOOKUP_SYNC_DELAY` 는 건너뛴 스텝엔 기록하지 않는다(왜곡 방지).
- 반환 계약 `(None, False)` 그대로 → 코어 스케줄러 쪽은 무변경.

## 2. 바꿀 것과 접근

### 2-1. 카운터는 CPU 티어에 둔다 (매니저 레시피 1 수정)

매니저는 `TieringOffloadingManager._complete_promotion` 에 승격 완료 카운터를 두자고 했다. 그러면 **승격이 아닌 원인의 HIT_PENDING** 이 단축에 갇힌다: GPU→CPU 저장이 진행 중인 블록도 `CPUOffloadingManager.lookup`(`cpu/manager.py:229-235`)은 `HIT_PENDING` 을 돌려주고, 그 해소는 `complete_store` 지 `_complete_promotion` 이 아니다. 승격 카운터만 보면 저장 대기 요청은 카운터가 안 바뀌어 영원히 재조회하지 않는다(행). 따라서:

- `CPUOffloadingManager` 에 `_state_epoch: int = 0` 을 두고, 조회 결과가 바뀔 수 있는 모든 전이에서 +1:
    - `complete_store`(성공·실패 모두, 실제로 바뀐 블록이 있을 때) — 저장 완료와 승격 완료(`complete_write` 별칭)를 한 곳에서 덮는다.
    - `prepare_store` 의 퇴거(`cpu/manager.py:327-350`, `to_evict` 가 비어 있지 않을 때) — 대기 요청의 이미-HIT 블록이 밀린 경우 다음 스텝에 정직하게 재조회하도록.
    - `reset_cache`.
- `OffloadingManager`(`vllm/v1/kv_offload/base.py`) 에 `state_epoch` 속성 추가, 기본 `None`(= 단축 비활성). 문서: "조회 결과가 바뀔 수 있을 때마다 증가하는 단조 카운터. `None` 이면 미지원".
- `TieringOffloadingManager.state_epoch` → `self.primary_tier.state_epoch` 위임. 2차 티어(fs)의 상태 변화는 RETRY 로 표현되므로 §2-4 로 걸러진다.

에포크가 같으면 "직전 스텝의 `None` 답이 그대로" 임이 정확히 성립한다: 그 사이 CPU 티어의 어떤 블록도 ready/제거/퇴거되지 않았고, 다른 요청의 `prepare_write` 로 MISS 키가 HIT_PENDING 이 되는 것은 이미 `None` 인 답을 바꾸지 못한다.

### 2-2. `RequestOffloadState`(`offloading/scheduler.py:343-372`) 필드 1개

- `pending_lookup_epoch: int | None = None` — 직전 `_lookup` 이 HIT_PENDING 만으로 `None` 을 돌려줬을 때의 `state_epoch`. `None` 이면 단축 없음.
- `num_computed_tokens` 변화 감지는 새 필드 없이 기존 `num_locally_computed_tokens` 를 덮어쓰기 전에 비교한다(선점 후 재진입·로컬 프리픽스 증가 시 조회 시작점이 달라지므로 재조회).

### 2-3. `get_num_new_matched_tokens` 단축

`transfer_jobs` 검사 뒤, `_lookup` 직전:

```python
skip = (not request.skip_reading_prefix_cache
        and req_status.pending_lookup_epoch is not None
        and num_computed_tokens == req_status.num_locally_computed_tokens
        and self.manager.state_epoch == req_status.pending_lookup_epoch)
```

- skip 이면 `num_hit_tokens = None`, `LOOKUP_SYNC_DELAY` 미기록, `deferred_lookup_start_time` 유지. `update_num_hit_chunks`·`_touch` 는 그대로 실행.
- skip 이 아니면 정상 `_lookup`. 결과가 `None` 이고 RETRY 를 안 봤으면(§2-4) `pending_lookup_epoch = self.manager.state_epoch`(**조회 뒤에** 읽는다 — `lookup` 첫머리의 `_maybe_process_finished_jobs` 가 에포크를 올릴 수 있으므로 조회 뒤 값이 "이 답의 근거 상태"다). 그 외(`None` 아님, RETRY, `state_epoch is None`)는 `pending_lookup_epoch = None`.
- `_maybe_observe_lookup_async_delay`(`:600-612`)에서 `deferred_lookup_start_time` 을 소비할 때 `pending_lookup_epoch = None` 도 함께. `reset_cache` 루프(`:1860-1868`)에서도 리셋.

### 2-4. RETRY 감지 (매니저 레시피 2 의 전달 방식 수정)

`_maximal_prefix_lookup`/`_sliding_window_lookup` 은 `req_status` 를 받지 않고, 테스트가 그 시그니처로 직접 호출·몽키패치한다(`test_scheduler.py:1005-1068` 의 `lambda keys, ctx, *_: 1`, `:1615` `_maximal_lookup`). 시그니처를 바꾸지 않기 위해 스케줄러 인스턴스 플래그 `self._lookup_saw_retry` 를 쓴다: `_lookup` 첫머리에서 `False`, 세 루프(`:651`, `:686`, partial-tail `:977`)의 RETRY 분기에서 `True`. 스케줄러는 단일 스레드이고 `_lookup` 은 한 요청씩 순차라 안전하다.

### 2-5. 영향 사이트

| 사이트 | 영향 |
| --- | --- |
| `vllm/v1/core/sched/scheduler.py` 코어 | 무변경(반환 계약 동일) |
| `OffloadingManager` 구현체: `CPUOffloadingManager`, `TieringOffloadingManager`, 테스트 mock(`MagicMock(spec=OffloadingManager)`, `utils.py:129`) | 베이스 기본 `None` 이라 미구현체는 단축 꺼짐. **mock 은 `state_epoch` 접근 시 MagicMock 을 돌려주므로** `RequestRunner`(`utils.py:129-133`)와 `_make_scheduler_with_lookup`(`test_scheduler.py:1594`)·`test_pending_transfer_defers_prefix_lookup`(`:2485`)에 `manager.state_epoch = None` 을 명시해 기존 테스트의 계약을 고정한다. |
| p2p/obj/fs 2차 티어 | 무변경(RETRY/HIT 로만 상호작용) |
| 메트릭 `LOOKUP_SYNC_DELAY` | 건너뛴 스텝 미기록 → 복원 중 호출 수가 요청당 1~2회로 줄어드는 것이 곧 검증 지표 |
| 디코드 중 다른 세션 | 대기 창(≈0.3~0.6 s)의 스텝 비용이 touch(≈5 ms)만 남음 |

### 2-6. 알려진 한계

- 대기 중 CPU 풀이 통째로 회전해 이 요청의 HIT 블록이 퇴거되면 §2-1 의 퇴거 에포크로 다음 스텝에 재조회한다. 승격 job 이 영영 안 끝나는 경우는 지금도 무한 폴링이라 새 실패 모드가 아니다.
- 파이썬만 바뀐다. 노드 `.so` 재빌드 없음.

## 3. 테스트

`tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` 의 deferred 묶음(`:517-583`) 옆에 4건(매니저 5절):

- (a) `lookup` 이 HIT_PENDING, `manager.state_epoch = 1`: 첫 스텝 이후 에포크가 그대로인 스텝에서 `lookup.call_count == 0`, `touch` 는 매 스텝 호출, `LOOKUP_SYNC_DELAY_count` 는 1.
- (b) `state_epoch = 2` 로 올리면 다음 스텝에 재조회; HIT 로 바꾸면 해소되며 `LOOKUP_ASYNC_DELAY_count == 1`.
- (c) RETRY 로 시작한 요청은 에포크가 같아도 매 스텝 `lookup` 호출.
- (d) 단축 중 abort → `on_request_finished` 경로 정상(`:585` 변형).
- 회귀 기준: 기존 deferred 3건, `test_concurrent_lookups_of_the_same_prefix`, `test_skip_reading_prefix_cache`, `test_request_preemption`.

`tests/v1/kv_offload/cpu/test_manager.py`: `complete_store` 성공/실패, 퇴거, `reset_cache` 각각에서 `state_epoch` 증가, 변화 없는 호출은 불변. `tests/v1/kv_offload/tiering/test_tiering_offloading.py`(`TestTieringOffloadingManager`): 승격 완료(`_process_finished_jobs`)가 `manager.state_epoch` 를 올린다.

실행은 헤드 venv, **서버 정지 창**(`FLASHINFER_DISABLE_VERSION_CHECK=1`, #26). 기준: #25 의 316 통과 + 신규. 로컬 macOS 는 torch 없어 `pre-commit run ruff-check`·mypy 만.

## 4. 검증(클러스터) — 한 단계씩, 30~60 s 모니터링, 전부 포그라운드

### 4-1. Step 0 실측(코드 전) 과 판정 규칙

계측 두 가지, 모두 기존 도구:

1. **조회 히스토그램 증가분**: 복원 전후 `/metrics` 의 `vllm:kv_offload_lookup_sync_delay_seconds_bucket{le=0.01|0.05|0.1|0.5}`·`_count`·`_sum` 차분. "10 ms 이상 조회 횟수"가 재조회 패스 수다(직접 지표).
2. **동시 세션 디코드**: `bench2.py <tag> --levels 1 --skip-public` 의 C1 tok/s(매니저 요구). 단, 복원당 얹히는 창이 0.3~0.6 s 라 수십 초 평균인 C1 tok/s 는 1% 미만만 움직인다. 그래서 스트리밍 토큰 간격을 직접 재는 작은 프로브 `deploy/gb10-cluster/dsv41/decode_gap_probe.py`(stdlib, ~60줄: 긴 디코드 1개를 스트림으로 받아 청크 간격의 최대·50 ms 초과 횟수·p99 를 JSON 으로)를 추가한다. 기존 `bench2.py` 가 청크 간격을 남기지 않아서다. 이 파일은 워크스테이션 전용이라 노드 반영 불필요.

순서(태그 `i27-*`, 원본은 헤드 `~/dsv41-prep/bench/i2-results.log` 뒤에 이어 씀, 요약은 `.notes/2026-09-12-issue-27-pending-lookup/results.md`):

1. 상태 확인: `dsv41_ctl.sh status`·`caps`, health 200, head `/proc/meminfo` MemAvailable ≥ 4.5 GiB, earlyoom 0. 지금 :8889 는 21:03 부팅 채택 구성이고 8K/32K/70K 만 거쳤으므로 P13·S2 는 fs 티어에만 있다(콜드).
2. 대조: `decode_gap_probe.py i27-ctl` + `bench2.py i27-ctl --levels 1 --skip-public`(복원 없음).
3. 복원 1건: gap 프로브 시작 → 3 s 뒤 `kvoff_probe.py i27-p13 P13 33000`(레포 사본, 헤드 `~/dsv41-prep` 사본은 구버전 #28) → 히스토그램 차분·gap·TTFT·hit 492,928·정답 기록. 같은 부팅에서 S2 로 1회 더(2표본).
4. 재기동(P13·S2 를 다시 콜드로) → 82K 웜업 → gap 프로브 + `kvoff_concurrent.py i27-c2 2 33000 P13,S2` → 같은 항목(hit 985,856).
5. **판정 — "노이즈 안"**: (i) 복원 1건당 10 ms 이상 조회 증가분이 2회 이하이고 (ii) 복원 중 최대 토큰 간격이 대조 최대 간격의 1.5배 이하이며 (iii) C1 tok/s 차이가 ±3% 안이면 코드 없이 마감(§0-2). 하나라도 벗어나면 구현. (i) 만으로도 재조회 패스 수는 드러나므로 (ii)(iii) 가 노이즈 안이라도 (i) 가 수십 회면 "관측됨"으로 본다 — 매니저 규칙("스텝 지연이 관측되지 않으면")의 구체화이며 오너 확인 항목 §6-1.

### 4-2. 코드 뒤

1. 워크트리 커밋 → `git format-patch` → 4대 `~/vllm-dsv41` 에 `git am`(서버 정지 창, #25 의 `stage`/`sync_nodes` 흐름을 `.notes/.../i27_ctl.sh` 로 재작성; `.so` 재빌드 없음).
2. `KV_TRACE=1` 부팅(≤ 170 s) → 웜업 → 4-1 의 3·4 반복. 채택 기준: 10 ms 이상 조회 증가분 요청당 1~2회, 복원 중 gap·C1 이 대조와 동급, 승격 job 0.32 s 대·TTFT 7 s 대·hit·정답이 #25 표와 같음.
3. 채택 구성(dsv41.env 기본값) 재기동 → 웜업 → `bench2.py i27f --levels 1,4`(기준 i25f 52.18/47.05, 34.00/121.01, ±3%) → `divergence.py record i27f` + `compare i25f i27f` → 새 salt 493K 콜드 프리필(`memlog.py`, head 바닥 ≥ 3.0 GiB, earlyoom 0; 시작 전 유휴 ≥ 4.5 GiB).
4. 종료 상태: :8889 health 200, 4대 `gpu-clock-cap` active·1989 MHz, head ≥ 4.5 GiB. README `KV offload host tier` 절에 측정 한 단락.

Step 0 에서 코드 없이 마감하는 경우에도 4-1-4 의 재기동 뒤 채택 구성 복구·종료 상태 확인은 동일하게 한다.

## 5. 이번에 하지 않을 것

- `HIT_PENDING` 에서 스캔을 break 하는 변경(첫 조회에서 승격 대상 키를 끝까지 등록해야 하므로 의미가 바뀜).
- 코어 스케줄러의 `step_skipped_waiting` 폴링 구조 변경, 승격 완료 콜백/이벤트 신호.
- 대기 중 `_touch` 생략(§1 불변 조건).
- 2차 티어(fs) RETRY 경로의 재조회 단축.

## 6. 오너 확인 사항(plan-approve)

1. §4-1-5 의 "노이즈 안" 구체 규칙(10 ms 이상 조회 ≤ 2회/복원, 최대 간격 ≤ 대조 1.5배, C1 ±3%) 으로 "코드 없이 마감" 을 판정해도 되는가. 매니저는 정성 규칙만 줬다.

- 전제(질문 아님): README 운영 규칙은 #20(글롭 삭제) 이 열려 있는 동안 워커의 `dsv41_ctl.sh start/stop` 실행을 금한다고 적혀 있으나, 이슈 #27 7절과 #25 실행(부팅 5회)은 재기동을 전제한다. #25 선례대로 `dsv41_ctl.sh start/stop` 을 쓴다.
- 전제: `decode_gap_probe.py` 는 측정 도구로 레포 `deploy/gb10-cluster/dsv41/` 에 커밋한다(C1 평균이 0.5 s 사건을 못 잡기 때문). 원치 않으면 히스토그램 차분 + C1 만으로 진행.
