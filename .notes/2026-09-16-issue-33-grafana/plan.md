# 이슈 #33 계획: ds4f-vllm 프리필 패널을 계산 토큰 기준으로 바꾸고 DS4F 전용 패널·규칙을 정리한다

작성 2026-09-16 KST, 워크트리 `issue-33-grafana`(이 레포, base `ae459a88f0`). 정본은 이슈 본문과 매니저 조사 `icmt-a582b124`(3~6 절)다. 계획 단계에서 한 일은 읽기 전용 확인뿐이다. homelab `origin/main` `2ae790c` 의 JSON·규칙을 읽었고, VM 쿼리를 보냈고, 헤드에서는 `casebench.py` 읽기와 `/metrics` 이름 목록 조회만 했다. 설정 변경과 요청 전송은 없었다.

## 1. 현재 동작과 문제점

정본은 homelab `k8s/apps/grafana/dashboards/ds4f-vllm.json` 이다(`origin/main` = 라이브, 패널 63 개, `version` 3, `jq --indent 1` 포맷). 이 파일은 ConfigMap `grafana-dashboards` 로 들어가고(`disableNameSuffixHash: true`) Flux 가 10 분 주기로 배포한다. 수집은 VM CT116 이 `:8888` 을 job `vllm`, `service=ds4f` 로 긁는다.

| 문제 | 패널 id | 현재 expr 의 결함 | 계획 단계 확인 |
| --- | --- | --- | --- |
| 프리필에 복원이 섞임 | 3, 42(A), 43 | `prompt_tokens − prefix_cache_hits − connector_prefix_cache_hits`. 마지막 이름은 없는 지표라 결과가 `local_compute + external_kv_transfer` 가 된다 | 헤드 `/metrics` 에 `connector_` 없음, `external_prefix_cache_hits_total` 과 `prompt_tokens_by_source_total{source=local_compute/external_kv_transfer/local_cache_hit}` 은 VM 에 시계열 1 개씩 있음 |
| 재프리필 패널이 복원을 계산으로 셈 | 40 | `prefix_cache_queries − prefix_cache_hits` | — |
| 스텝/초가 늘 0 | 12 | `ds4f_step_total{node="gx10-6040"}`(DS4F 도커 로그 tail 익스포터) | VM 에 4 노드 시계열은 있으나 값 0 |
| 감지기·가드 패널이 비어 있음 | 62, 63, 68 | `vllm:dspark_*` | VM 에 `vllm:dspark*` 0 개 |
| 정지 stat 오탐 | 17 | 진행 신호가 `prompt_tokens` rate == 0 이다. 긴 프리필 중에는 이 값이 0 이다. 실행 기준도 < 8 이다 | 실행 한도는 `SEQS=16` |
| 정지 시그니처 실행 기준 | 18 | 실행 기준이 < 8 이다(스텝 기준식은 이미 쓰고 있음) | 아래 1-1 |
| 대시보드 제목 | — | "DS4F vLLM — gx10" | — |
| vmalert 죽은 규칙 | `rules/ds4f.yml` | `DS4FDetectorFired`, `DS4FThinkingBudgetHits`, `DS4FLeakGuardStops` 가 `dspark_*` 를 봄 | 나머지 3 규칙(지문, runaway, long decode) 은 V4.1 지표를 씀 |

### 1-1. 매니저 표에서 한 가지 보강: 스텝 카운트만으로는 긴 프리필 중 정지 오탐이 남는다

`vllm:iteration_tokens_total` 은 요청 출력이 있는 스텝에서만 관측된다. `async_llm.py:778` 은 `IterationStats() if (log_stats and num_outputs) else None` 이다. 중간 프리필 청크는 출력을 내지 않는다. VM 에서 #32 의 128K 솔로 프리필(2026-09-16 01:00:45~01:02:00 KST, 실행 1, 대기 0)을 보면 30 초 창의 iteration 증가가 0~1 이었다. 그래서 매니저 식(대기 ≥1 · 실행 <16 · 스텝 rate <0.2)은 긴 프리필 뒤에 요청이 대기하면 여전히 정지로 뜬다. 대안으로 `vllm:estimated_flops_per_gpu_total` 은 있지만 MFU 가 꺼져 있어 늘 0 이다.

같은 창에서 `vllm:kv_cache_usage_perc` 는 스크레이프마다 올랐다(0.017 → 0.039, `changes[1m]` 2~4). 이 게이지는 출력 유무와 상관없이 매 스텝 오는 scheduler_stats 로 갱신된다(`scheduler.py:2362-2374`). 그래서 17·18 의 식에 "2 분간 KV 사용률 변화 없음" 항을 곱한다. 7 일 창(`sum_over_time(...[7d:1m])`)에서 정지로 뜬 분 수는 라이브 18 이 3, 매니저 식이 3, 보강식이 2 였다.

## 2. 바꿀 것

### 2-1. homelab `ds4f-vllm.json` (새 워크트리에서 손 편집, `jq --indent 1` 포맷 유지)

| id | 제목(변경 후) | expr | gridPos |
| --- | --- | --- | --- |
| 43 | 지금 프리필 tok/s | `sum(rate(vllm:prompt_tokens_by_source_total{job="vllm",source="local_compute"}[$__rate_interval])) or vector(0)` | 그대로 |
| 42 A | 실시간 프리필 vs 디코드 (tok/s) | 위와 같음(B 디코드는 그대로) | 그대로 |
| 3 | 프리필 속도, 계산 토큰 (t/s) | 위와 같음 | `x0 w12` → `x0 w8` |
| **72 신규** | 요청당 프리필 속도 (5분, 종료 요청 기준) | `sum(increase(vllm:request_prefill_kv_computed_tokens_sum{job="vllm"}[5m])) / sum(increase(vllm:request_prefill_time_seconds_sum{job="vllm"}[5m]))` | `x8 y8 w8 h8` |
| 2 | 디코드 속도 (t/s) | 그대로 | `x12 w12` → `x16 w8` |
| 40 | 복원 토큰/초 (디스크·CPU 티어) | `sum(rate(vllm:prompt_tokens_by_source_total{job="vllm",source="external_kv_transfer"}[$__rate_interval]))` | 그대로 |
| 12 | 출력 스텝/초 | `sum(rate(vllm:iteration_tokens_total_count{job="vllm"}[$__rate_interval]))` | 그대로 |
| 17 | 지금 상태 | `(sum(vllm:num_requests_waiting{job="vllm"}) >= bool 1) * (sum(vllm:num_requests_running{job="vllm"}) < bool 16) * (sum(rate(vllm:iteration_tokens_total_count{job="vllm"}[2m])) < bool 0.2) * (sum(changes(vllm:kv_cache_usage_perc{job="vllm"}[2m])) == bool 0)` | 그대로 |
| 18 | 정지 시그니처 (1=정지: 대기≥1, 실행<16, 출력 스텝≈0, KV 변화 없음) | 17 과 같음 | 그대로 |
| 62, 63, 68 | 제거 | — | — |
| 64 | 툴콜 파서 결과/5분 | 그대로 | `x16 w8` → `x0 w24`(빈 칸 메움) |
| 65, 66, 67 | 그대로 | 그대로 | `w6` → `w8`, x 는 0/8/16 |
| 19, 23 행 | "양 노드" → "4 노드" | 그대로(노드 필터 없는 `gb10_*` 4 노드) | 그대로 |
| 대시보드 | "DSv4.1F vLLM (gx10)" | uid `ds4f-vllm`, tag, job, `service=ds4f` 라벨은 그대로(시계열·링크 연속성) | — |

설명(`description`)은 바꾸는 패널에만 한두 문장씩 쓴다. 3·42·43 설명은 "첫 토큰이 나올 때 한꺼번에 기록되므로 긴 프리필은 톱니나 한 번의 스파이크로 보인다. 속도는 72 를 본다" 를 담는다. 72 설명은 "복원 요청은 복원 시간이 분모에 들어가 낮게 보인다" 를 담는다. 12 설명은 "프리필 청크만 도는 스텝은 세지 않는다" 를 담는다. 17·18 설명에는 1-1 의 KV 항 근거를 쓴다. 기존 설명의 DS4F 잔재("스텝 로그 실측 (issue43 -> node exporter)", "이 포크") 는 지운다. 수정 대상이 아닌 패널의 id·gridPos 는 그대로 둔다. 새 패널 id 는 현재 최대값 71 다음인 72 다.

### 2-2. homelab `proxmox/lxc/victoriametrics/rules/ds4f.yml`

`ds4f.agent-stall` 그룹에서 dspark 3 규칙을 지우고 3 규칙만 남긴다. 그룹 주석의 "(2) 새로 넣은 서버측 감지기 발동" 문구도 지운다. 남기는 규칙의 alert 이름(`DS4F*`)과 `domain: ds4f` 라벨은 Alertmanager 라우팅과 이력 연속성을 위해 그대로 둔다. `ds4f.memory` 그룹은 건드리지 않는다.

### 2-3. 이 레포

- `.notes/2026-09-16-issue-33-grafana/{plan.md,results.md}`. `.notes` 는 이미 추적 대상이라(`git check-ignore` 결과 없음) `.gitignore` 는 고치지 않는다.
- `deploy/gb10-cluster/dsv41/README.md` 의 `## Operating rules` 끝에 `### Metrics in Grafana (issue #33)` 한 문단을 둔다. 프리필은 `local_compute`, 복원은 `external_kv_transfer` 로 읽는다. 프롬프트 토큰은 첫 토큰 시점에 기록된다. 속도는 요청당 패널로 본다. 정지 판정에 KV 항을 쓴다. 헤드 `~/vmagent/scrape.yml` 은 죽은 파일이다.
- 코드 변경은 없다.

## 3. 영향 사이트

- Grafana: 소비자는 `ds4f-vllm` 대시보드 자체뿐이다. uid·tag 를 유지하므로 다른 대시보드의 링크는 그대로 산다. 제거하는 62/63/68 은 alert·링크 참조가 없다(`git grep dspark` 결과 `ds4f.yml` 뿐).
- vmalert: 지우는 3 규칙은 V4.1 에서 평가값이 없어 발동한 적이 없다. 남기는 규칙의 annotation 이 가리키는 대시보드 패널(위치별 수락률 57, 생성 토큰 2만 초과 60)은 그대로 있다.
- VM 수집(`prometheus.yml`)과 헤드 `~/vmagent/scrape.yml` 은 건드리지 않는다.
- 프로덕션 서버: 설정 변경과 재기동은 없다. 측정 요청만 보낸다.

## 4. 단계별 실행

### 4-1. implement

1. `git -C ~/workspace/nacyot/homelab worktree add ~/workspace/worktrees/nacyot/homelab/issue-33-dsv41-panels-20260916 -b issue-33-dsv41-panels origin/main` 으로 워크트리를 만든다. 로컬 `main` 과 그 미커밋 편집은 건드리지 않는다.
2. JSON 과 규칙을 편집한 뒤 정적 검사를 한다. JSON 은 `jq` 파싱, id 중복 0, gridPos 겹침 0(스크립트), 패널 수 61, `jq --indent 1` 왕복 동일을 본다. 바뀐 expr 은 VM `/api/v1/query` 로 전부 보내 오류 0 인지 본다. 규칙은 CT116 pinned vmalert 이미지에서 `-rule=<새 파일> -dryRun` 으로 검사한다.
3. 커밋 제목은 `grafana: count computed prefill tokens and drop DS4F-only panels on ds4f-vllm (#33)` 으로 한다. 최신 `origin/main` 위로 rebase 한 뒤 `git push origin HEAD:main`(fast-forward) 한다.
4. 배포를 확인한다. `ssh ser9 'kubectl -n flux-system get kustomization apps'` 의 revision 이 새 SHA 인지 본다. 그 뒤 `/api/dashboards/uid/ds4f-vllm` 의 title, 패널 수, expr 이 새 파일과 같은지 본다.
5. CT116 규칙을 바꾼다. 먼저 라이브 `/opt/victoriametrics/rules/ds4f.yml` 과 레포 원본을 diff 한다. 다르면 멈추고 기록한다. `ds4f.yml.bak-20260916-dsv41` 로 복사한 뒤 새 파일로 바꾸고 `docker kill --signal=HUP vmalert` 로 재적재한다. `/api/v1/rules` 에서 `ds4f.agent-stall` 규칙 3 개, `ds4f.memory` 2 개, lastError 0 을 확인한다.
6. 이 레포에 README 문단과 노트를 커밋한다.

### 4-2. validate (프로덕션 :8888, 재기동 없음, 한 단계씩)

1. 시작 상태를 기록한다. health 200, override 0 bytes, caps 4 대 1989, 헤드 MemAvailable 을 본다. 그다음 `systemctl --user stop dsv41-watchdog.timer` 로 워치독을 멈춘다.
2. 32K 프리필을 3 회 보낸다. 명령은 `~/sglang-cmp/casebench.py --mode solo --prefills 1 --prefill-tokens 32000 --tag i33-s32-r{1,2,3}` 이다. 프롬프트는 `random.Random(f"{tag}-{time.time()}")` 로 매번 새로 만들어 캐시에 맞지 않는다. 회차마다 jsonl 에 prefill 이 1 개인지 확인한다. 외부 트래픽 섞임은 창 안의 `request_success_total` 증가가 3 인지로 판정한다. 섞이면 그 회차를 다시 한다.
3. 판정. 기준 속도는 casebench `Σprompt_tokens / Σwall_s` 다.
   - (a) r3 종료 직후 72 패널 식(창이 r1~r3 을 모두 덮고 앞선 요청은 포함하지 않는 5 분)이 기준 속도와 10% 안이다. 이것이 이슈의 "5 분 평균이 실제 속도와 10% 안" 이다.
   - (b) 같은 창의 `increase(prompt_tokens_by_source_total{source="local_compute"})` 가 `Σprompt_tokens` 와 10% 안이다(계산 토큰 수 정확성). 43/3 의 rate 패널은 처리량이라 속도와 직접 같지 않다. 값은 기록만 한다.
   - (c) 세 회차 동안 17·18 의 식이 0 이다.
4. 복원 1 회. `dsv41_ctl.sh headroom` 이 5.2 GiB 이상이면 RC#4/#15 절차(`~/dsv41-prep/bench.py`)로 SSD 티어에 남은 493K salt 를 복원한다. 조건이 안 맞거나 저장소에 없으면 158K longctx 로 대체한다(콜드 1 회 → GPU 밀어내기 → 같은 프롬프트). 사유는 기록한다. 판정은 복원 창에서 `local_compute` 증가가 꼬리 수준(프롬프트의 1% 미만)이고 `external_kv_transfer` 증가가 로드 토큰과 맞는 것이다. `request_prefill_kv_computed_tokens_sum` 증가분도 기록한다. 코드상으로는 `stats.py:286` 의 cached 에 외부분이 들어가므로 꼬리 수준이 예상된다. 아니면 72 의 분자를 `local_compute` 증가로 바꾸고 재배포한다.
5. 라이브 대시보드 API 로 받은 모든 expr 을 VM 에 보내 오류 0 인지 본다. 1-1 의 7 일 정지 분 수를 다시 적는다.
6. 워치독을 `start` 로 되돌린다. 골든룰을 확인한다: health 200, override 0 bytes, `Environment` 비어 있음, caps 4 대 1989, `dsv41-watchdog.timer` active, `dsv41-ATTENTION` 없음, 노드 트리 clean, 헤드 MemAvailable 기록. 결과는 `results.md` 에 쓴다.

**merge**: 이 레포 브랜치를 로컬 `main` 에 머지한다(homelab 은 implement 에서 이미 반영).

## 5. 화면 변화

- 처리율 행이 3 칸이 된다: 프리필 속도(계산 토큰), 요청당 프리필 속도(신규), 디코드 속도. 복원 순간 프리필 선이 8K tok/s 로 치솟지 않고, 복원은 아래 "복원 토큰/초" 에 따로 보인다.
- "스케줄러 스텝/초" 가 "출력 스텝/초" 가 되어 0 대신 실제 값을 보인다.
- 에이전트 턴 정지 감시 행의 빈 패널 3 개가 사라진다. 툴콜 파서 결과가 행 전체 폭이 되고, 아래 stat 3 개가 1/3 폭씩 된다.
- 제목이 "DSv4.1F vLLM (gx10)" 가 되고, 시스템·GPU 행 제목이 "4 노드" 가 된다.

## 6. 이번에 하지 않을 것

- job/`service=ds4f` 라벨, uid, alert 이름 변경(시계열·라우팅 연속성).
- 헤드 `~/vmagent/scrape.yml` 삭제(기록만), 워크스테이션 `homelab-dashboard` 클론 정리.
- 프리필 청크 단위 실시간 속도(엔진 카운터 신설이 필요, 코드 변경).
- MFU 지표 켜기(서버 인자 변경 = 재기동).
- 대시보드 전체 재생성과 다른 패널 재배치.

## 7. 오너 결정

없음. 1-1 의 KV 항 보강은 측정 근거가 있는 기술 판단이라 계획에 넣었다. validate 의 (c) 와 7 일 재집계로 확인한다.
