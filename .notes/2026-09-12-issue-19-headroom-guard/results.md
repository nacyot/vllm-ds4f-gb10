# 이슈 #19 implementation results

## 구현 단계

- 기준 main: `d9b632b913b4a313d7800361f290762baab1a55b`.
- 승인 계획의 7개 파일만 수정했다. run_prompt의 시작 5.2 GiB 거부, 공용 2초 감시의 2.8 GiB 스트림 중단, 세 CLI의 JSON/exit 3, headroom/status, selftest 및 README를 구현했다.
- 실제 로컬 HTTP 서버에서 두 응답이 첫 SSE 토큰을 기다리는 동안 socket shutdown으로 모두 중단됨을 검증했다. urllib의 응답 헤더를 받은 뒤 감시를 등록한다. 헤더가 오기 전 HTTP 연결 대기는 기존 timeout을 유지한다.
- 기존 모델 발견/메트릭/tokenize 요청은 보존한다. 가드의 요청 없음은 chat/completions 추론 요청이 없다는 뜻이다.
- SPDX 헤더는 변경 파일에 프로젝트 pre-commit 훅이 추가했다. Python 환경은 uv venv --python 3.12 및 uv pip install -r requirements/lint.txt pytest로 준비했다. vLLM/torch 설치·import는 하지 않았다.

## 로컬 검증

- `.venv/bin/python -m pytest deploy/gb10-cluster/dsv41/test_headroom.py -v`: 13 passed (0.69 s).
- `/opt/homebrew/bin/bash -n deploy/gb10-cluster/dsv41/dsv41_ctl.sh deploy/gb10-cluster/dsv41/selftest_caps.sh`: exit 0.
- `/opt/homebrew/bin/bash deploy/gb10-cluster/dsv41/selftest_caps.sh`: 기존 캡 23개와 headroom 10개 케이스 통과. 모든 원격 명령은 스텁으로만 기록했다.
- selftest 산출물: `/var/folders/3l/vpffn08x3v92f1bly87zmjgh0000gn/T/tmp.tiZUiHBX9w` (삭제 없음).
- `.venv/bin/pre-commit run --files` 뒤 계획 범위 7개 파일: 적용 대상 훅 모두 통과. 초기 포맷/SPDX 자동수정과 shellcheck 변수 이름 충돌을 반영한 뒤 통과했다.
- `git diff --check`: 통과.

## validate 인계

클러스터 명령·배포·추론은 아직 실행하지 않았다. #27 바닥 ≥3.0 GiB 게이트와 클러스터 종료 상태는 아직 미검증이다. README 실측 표 다섯 번째 행은 실제 검증 후 추가한다. 계획서의 순서대로 웜 서버 음성 시험 → 신규 채택 부팅 → health → 82K 웜업 → headroom ≥5.2 → 신규 3글자 salt 493K 1회 → 반납/8K/종료 상태를 수행한다.

start/stop의 기존 /dev/shm 글롭 삭제는 실행 전에 실제 대상 확인과 명시 목록 처리가 필요하다. 캡 변경·kvfs 삭제·서버 노드의 pytest/torch 실행은 금지한다. 헤드 구 스크립트를 쓰지 않는다. 이슈 상태 변경은 매니저 소유다.

## validate 실측 — 493K 시작 게이트 미달

기준 `d9b632b913..HEAD` 차분을 원 요구사항·승인 계획·직접 소비자와 대조했다. 코드의 실질적인 추가 수정은 필요하지 않았다. 기준 ref 대상 pre-commit, Python 13개, Bash 33개를 재실행해 통과했다.

- 헤드 변경 전 SHA는 `908c821c3e3eeaf675bae598bef01e6ac6018cf4`. 로컬과 패치 적용 이력이 다르지만 세 Python 클라이언트의 blob은 기준과 일치했다. `git format-patch -1 6dc36b71ad --stdout -- <3 clients> | ssh ... git am`으로 헤드에 반영했다. 적용 후 `eea34808799cf18f0073c64623e79671e564d12f`. ctl은 워크스테이션에서 실행하므로 노드에 보내지 않았다.
- 노드의 uv 0.12.12 관리 Python 3.12.3 환경 `/home/nacyot/vllm-dsv41-venv`를, 존재하지 않음을 확인한 레포 `.venv`에 심볼릭 링크하여 재사용했다. pytest/torch import는 실행하지 않았다.
- 웜 서버: headroom 4.651951 GiB → exit 3. `i19-neg Nqz 33000`은 4.637241 GiB → skipped=headroom/exit 3. `prefix_cache_queries_total`은 전후 모두 82015로 추론 없음.
- 로컬 임시 디렉터리 `/tmp/i19-validation.25r0lc`에 원 ctl의 기존 rm 글롭 두 곳만 `:`으로 대체한 실행 사본을 만들었다. stop, 상태 확인, start를 각각 포그라운드로 실행했다. 시작 환경은 env -i에 HOME/PATH만 설정하여 설정 덮어쓰기를 배제했다.
- 정지 후 head `/dev/shm/vllm_offload_b8621a45-5dee-4677-ba28-f13216dc20dc.mmap`의 실제 경로·일반 파일·소유자 nacyot·fuser 사용 프로세스 없음·서버 inactive를 확인하고 이 파일 하나만 명시 삭제했다. 다음 관찰에서 네 노드 inactive, /dev/shm 비어 있음, 여유 116.81/117.55/117.53/117.54 GiB. kvfs 삭제와 안전 확인창 발생은 없었다.
- 신규 채택 TP=4 부팅: 2026-09-12 23:01:00 KST health 200, head 4.995 GiB.
- `i19-warm Wqz 5600`: prompt_tokens=82015, ttft_s=50.993, answer=12, ok=true. 23:01:55 head 로그: `empty_cache after prefill run of 81840 tokens: 2.93 GiB released`.
- 웜업 후 headroom: 4.610939 GiB → exit 3, 다른 노드 9.29/9.06/9.27 GiB. 지정된 5.2 GiB 미달로 493K 미실행. memlog 측정도 시작하지 않았고 README 다섯 번째 실측 행도 추가하지 않았다. #27 바닥 ≥3.0 GiB 게이트는 미완이다.
- 종료 상태 관찰: :8889 health 200, head 약 4.59 GiB, 네 gpu-clock-cap.service active, 직접 clocks.sm 조회는 전부 1989 MHz. 22:52 KST 이후 네 노드 earlyoom kill 이벤트 없음. status 5회 샘플에서는 한 번 27c4 최대 1995 MHz가 나왔고 이후 직접 유휴 조회에서 1989를 확인했다. 클록 조작 없음.
- #19 코멘트 `icmt-929be183`에 미달과 증거를 기록하고 지정대로 매니저 `tsk_165d7e23-3796-4719-aca1-ee16fee75fb2`에게 알렸다. 다음 지시까지 validate를 유지하며 merge로 진행하지 않는다. 이슈 상태는 변경하지 않았다.

## 매니저 판단 반영 및 validate 완료

`icmt-929be183`에 대한 매니저의 후속 지시에 따라 5.2 GiB 문턱을 유지하고 493K를 보내지 않는다. 신규 부팅 여유 부족은 가드가 막아야 할 상황으로 판단했으며, 추가 프리필·재기동·문턱 변경 없이 validate를 완료하고 merge로 진행하도록 승인받았다.

매니저가 확인한 비교 수치(워커가 추가 측정한 값이 아님):

| 항목 | #25 S4 | 이번 신규 부팅 |
| --- | ---: | ---: |
| head 유휴 MemAvailable | 7.11 GiB | 4.61 GiB |
| MemFree | 5.1 GiB | 1.6 GiB |
| swap 사용 | 1.5 GiB | 거의 미사용 |
| AnonPages | 3.0 GiB | 4.7 GiB |
| Worker_TP0 RssAnon | 2.17 GiB | 2.92 GiB |

유휴 여유는 약 2.5 GiB 감소했고, swap 사용 차이 1.5 GiB와 AnonPages 증가 1.7 GiB가 관찰됐다. 신규 부팅 여유는 4.6~7.1 GiB로 편차가 있어 5.2 GiB 이상을 전제할 수 없다. #25 S4의 유휴 7.11 GiB는 기존 표의 프리필 시작 6.95 GiB와 관측 시점이 다르다.

README 실측 표 다섯 번째 행에 이번 거부 결과와 비교를 추가하고, 신규 부팅 후에도 headroom 확인이 필수이며 5.2 미만이면 시작하지 않는다는 운영 규칙을 명시했다. **#27 바닥 게이트는 이 작업에서 미완이며, 후속 이슈는 매니저가 등록한다.** 기존 계획의 이 게이트 완료 요구는 이번 매니저 지시에 따라 후속으로 이관한다.

이번 후속 처리에서는 문서만 수정했다. 앞서 확인한 종료 상태(health 200, 4대 캡 active·유휴 1989 MHz, head 약 4.59 GiB)를 변경하는 클러스터 작업은 하지 않았다. 이슈 done은 매니저 소유로 유지한다.
