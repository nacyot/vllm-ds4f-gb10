# 이슈 #19 — 긴 프리필 head 여유 가드 계획

기준: main `d9b632b913b4a313d7800361f290762baab1a55b`(현재 HEAD 일치).
요구사항과 운영·검증 순서의 근거: #19 본문, 오너 결정 `icmt-81af5064`, 매니저 조사 `icmt-15ef874f`.
이 단계에서는 코드·클러스터·이슈 상태를 변경하지 않는다. plan-approve 승인 후 워크플로 단계에 따라 구현·검증·머지한다.

## 현재 동작과 문제, 유지할 조건

- `kvoff_probe.run_prompt`는 메모리 점검 없이 프롬프트를 만들어 `/v1/chat/completions`에 보내고 SSE를 읽는다. `kvoff_concurrent.py`의 ThreadPoolExecutor와 `prefill_probe.py`도 이 함수를 호출한다.
- 반환값은 salt, prompt_tokens, ttft_s, answer, expected, ok이며 각 CLI가 메트릭 차이·처리량·집계를 덧붙인다. 기존 정상 결과와 프롬프트 내용, 기본 max_tokens, 정답 판정은 유지한다.
- 기본 base는 head의 localhost:8889이다. prefill CLI는 base를 원격 주소로 바꿀 수 있다. 현재 모듈 import가 `/v1/models`를 조회하므로 단위 테스트는 import 이전부터 HTTP를 대체한다. 가드의 “요청 없음”은 추론 요청 없음이며 모델 조회·메트릭·tokenize와 구별한다.
- ctl의 `caps`/`start`는 관측 실패 시 exit 3을 사용하는 기존 관례다. `status`의 `free -g`는 정수 GiB라 문턱 확인에 부적합하다. 시작 순서와 캡 게이트는 유지하며 부팅에 headroom 조건을 추가하지 않는다.
- 과거의 4.5 GiB 시작 규칙은 #27에서 실패했다. 매니저가 정리한 약 2.1~2.2 GiB 하락을 고려해 5.2 GiB를 시작 기준으로 사용한다. 4.5 GiB는 종료 상태 기준으로만 유지한다. `dsv41.env`의 EMPTY_CACHE=1 및 최소 65536 토큰 정책은 의도된 결정으로 보존한다. earlyoom 2.43 GiB도 변경하지 않는다.

## 바꿀 것과 접근

수정 스코프는 `deploy/gb10-cluster/dsv41/` 아래 다음 7개 파일로 한정한다.

1. **kvoff_probe.py**: vllm/torch import 없이 `/proc/meminfo`의 MemAvailable을 GiB로 읽는다. 비 Linux·파일 부재·원격 base는 None과 경고 한 줄로 가드를 건너뛴다. URL의 hostname으로 localhost/127.0.0.1을 판별한다.
   - 모듈 환경 설정: `DSV41_MIN_AVAIL_GIB=5.2`, `DSV41_ABORT_BELOW_GIB=2.8`, `DSV41_LONG_PROMPT_RECORDS=18000`. 0인 문턱은 해당 가드를 끄며 records=0은 긴 프롬프트 가드 전체를 끄는 의미로 명시한다.
   - nrec가 records 문턱 이상이면 추론 요청 전 점검한다. 시작값이 5.2 미만이면 현재값·문턱·재기동 안내를 담은 HeadroomError를 발생시킨다. 정확히 문턱인 값은 허용한다.
   - 긴 요청만 모듈 전역 데몬 감시 스레드 하나에 등록한다. 잠금으로 활성 요청과 참조 수를 관리하고 2초마다 측정한다. 2.8 미만이면 등록된 긴 요청에 중단을 표시하고 연결을 끊는다. 각 요청의 시작값·최소값은 별도로 보존하고 finally에서 등록 해제한다.
   - 블로킹 SSE 읽기 중에도 취소가 즉시 전달되도록 응답 close 또는 socket shutdown을 사용한다. 첫 토큰 전 대기, 중단에 따른 읽기 예외, 정상 종료와 감시의 경합을 처리한다. 단순 close가 읽기 잠금에 막히면 shutdown으로 해소하며 이 부분만 최소 로컬 검증한다.
   - 정상/중단 결과에 `mem_avail_start_gib`, `mem_avail_min_gib`를 추가하고 관측 불가 값은 null로 표현한다. 중단 결과에는 `aborted: "headroom"`을 추가하고 성공으로 판정하지 않는다.
2. **kvoff_concurrent.py / prefill_probe.py**: HeadroomError를 잡아 tag, skipped=headroom, mem_avail_gib, min_gib의 JSON 한 줄과 exit 3을 반환한다. kvoff_probe.main도 동일하다. 동시 호출은 공용 감시를 사용하며 중단 결과를 정상 처리량·성공으로 오인하지 않게 소비 지점을 점검한다. 짧은 요청의 동작은 유지한다.
3. **dsv41_ctl.sh**: `headroom [min_gib]`를 추가한다. 인자 > MIN_AVAIL_GIB > 5.2 순으로 문턱을 정하고 4노드의 HOST/MEMAVAIL_GIB를 소수 둘째 자리로 표시한다. head 미달 또는 head 측정 누락·비수치·접속 실패는 exit 3과 head 이름·restart 안내를 출력한다. 비교에 앞서 입력을 검증한다. `DSV41_MEM_FIXTURE`의 host/gib 행으로 SSH 없는 검증을 지원한다. `status`도 meminfo 기반 소수 표시로 바꾸고 헤더/usage를 갱신한다.
4. **selftest_caps.sh**: 기존 ssh 스텁·fixture·run_case를 확장한다. status 스텁을 meminfo에 맞추고 기존 캡/시작 순서 테스트를 보존한다.
5. **test_headroom.py (신규)**: 인근에 Python 가드 테스트가 없으므로 이 스코프에 최소 pytest 테스트를 둔다. 아래 계약을 HTTP·메모리·시간 대체로 검증하며 실제 서버나 torch에 의존하지 않는다.
6. **README.md**: 시작 5.2 / 중단 2.8 / 종료 4.5를 구분한다. 기록 수 18000 기준과 대략적인 토큰 환산을 설명해 256K 토큰이라는 운영 지침과 정확한 토큰 계수 가드를 혼동하지 않게 한다. 복원 세션 후에는 대개 재기동이 필요함을 명시한다. headroom, 환경 설정·0 해제, 원격 예외, 로컬 테스트 명령을 추가한다. 기존 4행 실측 표를 싣고 검증 후 이번 실측을 다섯 번째 행으로 추가한다. status 정수 우회 안내와 클라이언트 변경에도 “노드 배포 불필요”로 읽히는 문구를 정정한다.

## 영향 사이트와 검증 설계

모듈의 목적은 프리필/복원 요청과 정답·시간 측정이다. 입력 계약은 salt/nrec/max_tokens, 출력 계약은 기존 결과 dict에 메모리 관측·중단 상태를 더하는 것이다. 방지할 실패는 낮은 head 여유에서 추론 시작, 읽기 대기 중 감시 중단 실패, 동시 호출의 감시 누수다. 가장 저렴한 검증은 가짜 응답을 사용하는 로컬 단위 테스트와 SSH 스텁 selftest다.

- Python: 긴 4.5 GiB 요청은 HeadroomError이고 추론 HTTP 미호출; 짧은 700 records는 진행; 5.2 경계 허용; 감시값 5.0→2.7은 시작 가드를 테스트 설정으로 허용한 뒤 응답 취소와 aborted·최소값을 확인한다. 원격 base는 경고 후 건너뛰고 0 해제도 확인한다. 두 긴 호출의 공용 감시와 종료 후 해제를 검증한다. 세 CLI의 거부 JSON/exit 3도 HTTP 스텁으로 확인한다. 첫 토큰 전 블로킹 읽기 취소는 실제 HTTP 연결을 쓰는 최소 로컬 확인이 필요할 때만 추가한다.
- Bash: head 6.95 통과, 4.52 거부와 gx10-6040/restart 안내, MIN_AVAIL_GIB=4.5 허용, 명시 인자 우선순위, 5.2 경계, head 누락/비수치/중복·잘못된 문턱 거부, fixture 실행의 SSH 미호출. 기존 caps/status 회귀 확인.
- 워크스테이션에서 Bash >=4로 `bash -n`, `selftest_caps.sh`; uv로 관리한 `.venv/bin/python -m pytest deploy/gb10-cluster/dsv41/test_headroom.py -v`; 프로젝트 지침의 pre-commit 설치·변경 파일 대상 관련 훅을 실행한다. 서버가 살아 있는 노드에서는 pytest/torch를 실행하지 않는다.
- 추가 JSON 필드는 기존 키를 보존한다. prefill의 처리량/중앙값 집계와 concurrent의 requests 배열이 직접 영향받는다. 지시에서 언급한 bench2_summary.py 및 #25 원시 메모리 로그는 현재 워크트리에 없어 내용 확인을 주장하지 않는다. 실제 검증에 필요하면 레포 사본·기존 측정 산출물 위치만 확인하며 ~/dsv41-prep의 구 스크립트를 실행하지 않는다.

## 클러스터 검증 — 지정 순서를 유지

각 클러스터 명령은 한 단계씩 포그라운드로 실행한다. 단계 간 및 장시간 실행 중 30~60초 간격으로 관찰한다. 백그라운드 잡 체인을 만들지 않으며, 지정된 1초 memlog systemd 유닛만 측정용으로 사용한다.

1. 로컬 검증 통과 후 `git format-patch | git am`으로 필요한 노드 레포에 반영한다. scp·부팅 중 파일 교체·.so 재빌드는 하지 않는다. Python은 노드에서도 `.venv/bin/python`을 사용한다.
2. 웜 서버에서 headroom 거부(exit 3)를 확인한다. 실제 값이 문턱 미만임을 먼저 확인하고 `i19-neg`, 새 3글자 salt, 33000 records로 추론 없는 exit 3만 검증한다. prefix_cache_queries 전후 불변을 기록한다. 웜 서버 여유가 예상보다 높아 음성 시험 전제가 성립하지 않으면 실제 긴 요청을 보내지 않고 값과 차이를 매니저에게 보고한다.
3. stop → 채택 dsv41.env 기본 TP=4 start(캡 게이트 유지) → :8889 health 200 → 5600 records의 82K 웜업 1회를 각각 실행·관찰·기록한다.
4. headroom의 head ≥5.2 GiB를 재확인한다. 미달이면 493K를 시작하지 않고 값을 이슈에 기록하여 매니저에게 알린다.
5. #25 방식의 memlog.py를 1초 systemd-run 유닛으로 가동한다. 신규 3글자 salt, 33000 records로 493,015 토큰 콜드 프리필을 단 1회 실행한다. 4글자 salt는 길이 초과하므로 금지한다. 실행 중 head 메모리·health·earlyoom을 30초마다 확인한다. 완주·정답·바닥 ≥3.0 GiB·earlyoom 0을 게이트로 삼고 예상 380~400초와 실제 시간을 함께 기록한다. JSON 2초 최소값과 memlog 1초 최소값 모두 ≥3.0인지 확인한다.
6. EMPTY_CACHE 반납 로그 1건·유휴 회복값을 기록한 뒤 8K 프로브 1회로 정상 동작을 확인한다.
7. 종료 시 채택 구성 :8889 health 200, 4대 gpu-clock-cap.service active·유휴 SM 1989 MHz, head ≥4.5 GiB, 작업 구간 earlyoom 0을 확인한다. 필요하면 채택 구성으로 복구하고 재확인한다. earlyoom 개입 시 즉시 실험 중단·채택 구성 복구·이슈 기록 후 설계를 재검토하며 게이트 통과로 보고하지 않는다.

현재 start/stop 구현에는 /dev/shm 글롭 삭제가 포함되어 있다. 검증 단계에서 해당 경로를 무검토로 실행하지 않는다. 먼저 노드별 실제 대상과 소유·작업 범위를 확인하고, 삭제가 필요하면 확인한 명시 목록만 사용하는 안전한 단계로 구성한다. kvfs 디렉터리는 삭제하지 않는다. 새 임시 산출물은 mktemp -d를 사용하고 안전 확인창 발생 시 대상 검증·판단·재개 결과를 기록한다. 캡 변경 명령과 clock_ctl.sh cap은 실행하지 않는다.

## 범위 밖과 승인 사항

엔진/vllm 코드, earlyoom 한계, dsv41.env 기본값, 클록/서비스 설정, 메모리 정책, KV 스토어는 바꾸지 않는다. 실기에서 2.8 GiB 중단을 일부러 유발하거나 웜 서버에서 493K 콜드 프리필을 실행하지 않는다. UI 변경은 없다.

오너만 결정할 미해결 설계 질문은 현재 없다. 위 계획을 plan-approve에서 검토받는다. 검증 결과는 이후 같은 디렉터리 results.md에 기록하고 README 표를 실측으로 갱신한다. #27 미완 게이트는 증거가 모두 충족된 경우에만 완료로 보고하며 이슈 in_progress/done 변경은 매니저가 맡는다.
