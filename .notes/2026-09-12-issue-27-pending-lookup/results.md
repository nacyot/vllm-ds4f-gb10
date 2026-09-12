# 이슈 #27 결과 기록 (2026-09-12) — 승격 대기 재조회 단축

원본 로그: 이 디렉터리의 `results.log`, 헤드 `~/dsv41-prep/bench/gap-i27-*.json`(청크 간격 원본), `~/dsv41-prep/logs/dsv41-r0.log`(`fs promotion job` 줄).

## Step 0 — 코드 전 실측 (21:32–21:42 KST, main caadeffc95 + 측정 프로브만)

계측: `decode_gap_probe.py`(헤드에서 1000 토큰 카운팅 디코드 1스트림, 청크 간격 = 스텝 시간; dspark k=5 라 청크당 5~6 토큰, 정상 스텝 ≈ 71 ms) + 같은 창의 `/metrics` 조회 히스토그램 차분. 복원은 프로브 시작 3 s 뒤 `kvoff_probe.py`/`kvoff_concurrent.py`(레포 사본).

| 케이스 | 디코드 tok/s | 간격 중앙값 | 2×중앙값 초과 스톨 | 최대 간격 | 10 ms+ 조회 | 50 ms+ 조회 | 조회 합 | 복원 TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 대조(복원 없음, i27-ctl1) | 84.1 | 71.2 ms | 0 | 78.8 ms | 0 | 0 | 0 | — |
| P13 493K 콜드 복원 1건 (i27-r1) | 62.8 | 70.8 ms | 6 | 1,763 ms | 9 | 4 | 1.81 s | 6.77 s |
| S2 493K 콜드 복원 1건 (i27-r2) | 60.8 | 70.9 ms | 6 | 1,855 ms | 9 | 4 | 2.05 s | 7.12 s |
| P13+S2 동시 복원 (i27-b1-c2, 재기동·KV_TRACE=1) | 47.5 | 71.9 ms | 10 | 1,796 ms | 13 | 10 | 4.56 s | 11.41 / 10.85 s |

조회 히스토그램 버킷(복원 1건당, 프로브 자신의 조회 1회 제외): 10~50 ms **5회**, 100~500 ms 3회, 500 ms~1 s 1회. 동시 2건: 10~50 ms 3회, 100~500 ms 8회, 500 ms~1 s 2회.

해석:

- 승격 대기(HIT_PENDING) 재조회는 승격 job 0.32 s 동안 스텝마다 1회 ≈ **5회 × 10~50 ms** — 매니저 추정(11.6K 키 25~60 ms)과 일치. 이것이 #27 의 대상이며 복원 1건당 약 0.1~0.25 s.
- 나머지 조회 4회(100 ms~1 s)는 요청당 1회씩인 첫 조회(RETRY, fs 비동기 조회 등록), 승격 개시 조회(prepare_write 11.6K), 최종 HIT 조회(lookup 이벤트 11.6K 기록) 등으로 **이번 스코프 밖**.
- 디코드 스톨 6회(합 ≈ 4.5 s) 중 큰 것(1.8 s @ 첫 토큰 직전, 0.6 s 직후)은 복원 요청 자신의 프리필·493K 컨텍스트 디코드 스텝이 같은 배치에 들어가는 비용으로 조회와 무관. 즉 #27 코드가 줄일 수 있는 몫은 복원당 스톨의 **5% 안팎**이다.
- 판정: 계획 §4-1-5 (i) 10 ms+ 조회 9회/복원 > 2 → "관측됨" → 구현 진행. (ii)(iii) 도 대조 밖(스톨 6회, tok/s −25%)이지만 그 대부분은 위와 같이 다른 원인.

## implement 단계 (21:00–21:55 KST)

- 커밋(워크트리 `issue-27-pending-lookup`, main caadeffc95 위): `0f54f33d57` 측정 프로브(`decode_gap_probe.py`), `78d3d536d1` 코드(base/cpu/tiering state_epoch + 커넥터 단축 + mypy 가드), `<tests>` 테스트.
- 노드 반영: 헤드는 코드만(`git am` 배치 i27-214541, 프로브는 이미 있음), 워커 3대는 프로브+코드(i27-214542). 4대 `.so` 재빌드 없음(파이썬만). 서버 정지 창.
- 단위 테스트(헤드 venv, 서버 정지, `FLASHINFER_DISABLE_VERSION_CHECK=1`): `cpu/test_manager.py` + `tiering/test_tiering_offloading.py` + `tiering/test_async_lookup.py` + `tiering/test_fs_tier.py` + `offloading_connector/test_scheduler.py` = **353 통과**(#25 기준 316 + 신규 5건 + 그 사이 상류 증가분). 141 s.
- 스모크 부팅 `KV_TRACE=1`(b2): health 200 in 149 s, fs_io_C 폴백 경고 없음, 82K 웜업 정답. head 유휴 7.66 GiB.

### 코드 후 단일 복원 (b2, KV_TRACE=1)

| 케이스 | 승격 job | 복원 TTFT | 디코드 tok/s | 10~50 ms 조회 | 100 ms~1 s 조회 | hit·정답 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| P13 (i27-b2-r1) | 0.343 s | 8.19 s | 68.4 | **1** (코드 전 5) | 4 (범위 밖) | 492,928 · 12 |
| S2 (i27-b2-r2) | — | 8.17 s | 58.0 | **1** | 4 | 492,928 · 12 |

- **핵심 지표**: HIT_PENDING 재조회가 사는 10~50 ms 버킷이 복원당 **5 → 1**. 남은 큰 조회 4회(100 ms~1 s)는 첫 조회(RETRY)·승격 개시(prepare_write 11.6K)·최종 HIT(lookup 이벤트 11.6K) 로 이번 스코프 밖이며 코드 전후 동일. 매니저 게이트의 "복원 중 10 ms+ 증가분 요청당 1~2회(첫 조회·최종 HIT)"는 재조회 몫에 한해 충족(5→1).
- 디코드 스톨(최대 ≈ 1.7~1.9 s, 8회)과 tok/s 는 코드 전과 사실상 동일 — Step 0 분석대로 스톨의 대부분은 복원 요청 자신의 프리필·493K 디코드 스텝(배치 동반)이라 #27 코드가 줄일 몫이 아니다. 즉 이 변경의 효과는 **동시 디코더가 겪는 스텝당 파이썬 재조회 부하 제거**(스텝 CPU 시간)이며, 벽시계 스톨(GPU 릴레이·프리필 지배)은 그대로다.
- 정확성 불변: TTFT 7~8 s 대, hit 492,928, 정답 "12", 승격 job 0.34 s 대 모두 #25 표와 같음. earlyoom 0.

### validate 로 넘기는 것

채택 구성(KV_TRACE 없음) 재기동 → 동시 복원(P13+S2) 히스토그램·gap 1회 → bench2 C1/C4(기준 i25f 52.18/47.05, 34.00/121.01 ±3%) → divergence record+compare(i25f) → 새 salt 493K 콜드 프리필(head 바닥 ≥ 3.0 GiB) → README `KV offload host tier` 절 측정 단락. 종료 상태 확인.

### 코드 후 동시 복원 (b3, 채택 구성, KV_TRACE 없음)

| 케이스 | wall | 복원 TTFT | 디코드 tok/s | 10~50 ms 조회 | hit·정답 |
| --- | ---: | ---: | ---: | ---: | --- |
| P13+S2 (i27-b3-c2) | 12.24 s | 12.19 / 11.65 s | 46.7 (코드 전 47.5) | **2** (코드 전 3) | 985,856 · 12 |

- 동시 재조회 몫은 단일보다 작게 준다(밴드 3→2): 두 세션의 첫 조회·프리필이 배치를 지배하고 승격 창(0.32 s×2)의 겹침이 짧기 때문. 단일 복원의 5→1 이 더 깨끗한 시연이다.
- 디코드 tok/s·TTFT·hit 는 코드 전과 동일 — 정확성·성능 회귀 없음.

## validate 단계 (22:05–22:20 KST, 채택 구성 :8889)

리뷰(메인 직접): `git diff caadeffc95..HEAD` 대조. 목표 충족(HIT_PENDING-only 대기 재조회 생략, touch·hit-chunk 유지, RETRY 는 단축 제외, epoch 은 lookup 뒤 읽기). 최초 요청(## 요구사항) 충족(Step 0 측정 → 노이즈 밖 → 코드, touch 유지). 회귀 없음(base `state_epoch=None` 로 타 매니저 무영향, mock `state_epoch=None` 고정). 발견한 데코레이터 오배치(`state_epoch`/`has_pending_work` 의 `@override`)를 그 자리에서 수정 커밋(`0552479e2e`, 런타임 무변경).

- 승격 완료가 skip 스텝에도 반영되는지 확인: `TieringOffloadingManager.on_schedule_end` 이 매 스텝 끝 `_maybe_process_finished_jobs()` 로 완료 job 을 처리(lookup 호출과 무관) → `complete_write`(=`complete_store`)가 epoch +1 → 다음 스텝 재조회. `primary_tier.complete_write`/`prepare_write` 는 CPU 매니저의 `complete_store`/`prepare_store` 별칭이라 계측 지점을 그대로 탄다.

### 게이트 결과

| 게이트 | 기준 | 결과 | 판정 |
| --- | --- | --- | --- |
| 단위 테스트(변경 범위) | 전부 통과 | 353 통과(cpu/tiering/async_lookup/fs_tier/offloading_connector) | 통과 |
| bench2 C1 (i27f vs i25f, 평균) | ±3% | agg +2.1%, per-stream +2.0% | 통과 |
| bench2 C4 (평균) | ±3% | agg −0.4%, per-stream −1.2% | 통과 |
| bench2 pi 6종 greedy | ok | 전부 ok=True | 통과 |
| divergence greedy (compare i25f i27f) | 동일 | ko-food·code·count 동일, ko-busan 토큰 24 argmax 수치편차(#25 와 동일 현상) | 통과(편차 기록) |
| 단일/동시 복원 정확성 | hit·정답 불변 | hit 492,928 / 985,856, 정답 "12", 승격 job 0.34 s, TTFT 7~8 s | 통과 |
| 재조회 절감(핵심 목표) | 재조회 감소 | 단일 10~50 ms 조회 5→1, 동시 3→2 | 통과 |
| 493K 콜드 프리필 head 바닥 ≥ 3.0 GiB | 바닥 ≥ 3.0, earlyoom 0 | **미완료** — 아래 참조 | 미완료(비적용) |

bench2 per-category C1 이상치(narrative +23%, coding +9%)는 C1 단일 배치(200 토큰, wall ~3.5 s) 소표본 편차다. 이 변경은 복원이 없는 정상 디코드에는 개입하지 않아 정상상태 성능에 영향이 없고, 게이트 기준인 C1/C4 평균은 ±3% 안이다.

### earlyoom 이벤트 기록 (안전 규칙)

- 22:14:10 KST, gx10-6040(head). earlyoom 이 `VLLM::Worker_TP`(pid 3510700, VmRSS 7216 MiB, badness 834)에 SIGTERM → 헤드 EngineCore 사망(health 000, GPU 유휴 208 MHz 다운클럭; cap 서비스는 계속 active).
- 원인: **테스트 방법 오류**. 앞선 동시 복원(P13+S2)이 CPU 티어에 상주한 상태의 웜 서버(head 유휴 4.52 GiB)에서 새 salt 493K 콜드 프리필을 시작. 프리필의 스토어 버퍼+워킹셋이 head MemAvailable 을 2.92 GiB(30 s 샘플)→earlyoom 임계 2.43 GiB 아래로 끌어내렸다. 코드 변경과 무관(스케줄러 조회 생략은 프리필 메모리 경로에 개입하지 않음). (직전 시도는 salt "S27a" 가 526,010 토큰으로 MAXLEN 524,288 초과 → HTTP 400, 크래시 아님.)
- 복구: 전체 정지 → 채택 구성 재기동(health 200, 150 s) → 82K 웜업. 골든룰 준수.

### 493K 콜드 프리필 게이트 — 미완료 사유와 비적용 근거

- 재기동+웜업 후 head 유휴 **4.59 GiB**(≥ 4.5 시작 게이트는 충족하나 여유가 얕음). 크래시 때 프리필이 head 를 약 1.6 GiB 끌어내린 실측을 감안하면 4.59 에서 재시도 시 바닥이 ~3.0 GiB 경계 또는 그 아래로 내려가 **2차 earlyoom·프로덕션 다운** 위험이 크다. `KV budget: no side effects`·소단계 원칙에 따라 재시도하지 않았다.
- **비적용 근거**: 이 변경은 `RequestOffloadState` 에 int 1개(`pending_lookup_epoch`), `CPUOffloadingManager` 에 int 1개(`_state_epoch`)를 더하고 할당 경로를 바꾸지 않는다. 493K 콜드 프리필의 메모리 바닥은 이 변경이 영향을 줄 수 있는 인과 경로에 없다(#25 는 fs 승격 C 확장·job 분할을 바꿔 이 게이트가 필요했음). 따라서 미완료지만 회귀 위험 0 으로 판단해 머지를 막지 않는다. 후속으로 head 유휴가 넉넉한 신규 부팅에서 확인 권장.
