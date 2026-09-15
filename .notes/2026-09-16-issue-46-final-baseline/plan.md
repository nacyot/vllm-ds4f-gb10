# #46 프로덕션 최종 기준선 측정 계획

## 현재 동작과 문제

- 기준 커밋은 `main 7c195a682797d5782e4efb26815f672ffb1e613d`다. 현재 로컬 HEAD가 일치하고 작업 트리는 깨끗하다. 노드의 엔진 동등성은 실행 직전 다시 확인한다. 이슈에서 설명한 README 차이는 배포 변경 사유가 아니다.
- #46 본문과 코멘트 `icmt-351f8dc6`가 실행 정본이다. 기존 `full-combo` 이후 채택된 기본값을 모두 포함하는 동일 회차 전체 측정이 없어, 기존 비교 자료에 `prod-final` 기준선을 추가한다. 엔진이나 벤치 도구 개선 작업은 아니다.
- head `gx10-6040:~/sglang-cmp/run_suite.sh`, `report_tables.py`, `~/dsv41-prep/i34/mixlong_i34.sh`를 읽었다. suite는 기본 웜업→디코드→프리필→혼합→에이전트→HOL→긴 세션 순서이며 JSONL에 append한다. casebench는 head 메모리를 자체 샘플링한다.
- **셀 수 표기 차이:** 실제 suite는 warm, warm2, decode, S8, S32, S128, M3, AG4r1, AG4r2, HOL, longctx의 11회 호출이다. 레시피의 S128 추가 2회와 MIXLONG 1회를 합쳐 전경 명령 14회다. MIXLONG은 내부에서 MX4/MXL 두 행을 만든다. 따라서 요구된 casebench 13행, decode 8행, longctx 1행과 일치한다. “12셀”을 맞추려 임의 셀을 추가하지 않는다.
- #32 결과의 타이머 stop/start 관례와 #34 결과의 HOL 7.01/8.71 s, MIXLONG 265.4 s를 확인했다. 표 도구는 일부 지표에 최초 일치 행을 쓰므로 태그 충돌과 재시도를 따로 관리해야 한다. 계획 조사 시 기존 JSONL에 prod-final 문자열은 없었다.

## 불변 조건과 영향 사이트

- 실행 대상: 6040 head의 기존 `:8888` 서버와 f323/37cc/27c4 워커. 4 노드 serve active, override 0 bytes, gpu-clock-cap active 및 SM 1989 MHz, head health 200, ATTENTION 없음이 기준이다.
- 노브·서버·클럭·코드·기존 결과는 바꾸지 않는다. head watchdog 타이머만 측정 동안 정지하고 모든 종료 경로에서 재개한다. kvfs-gc 등 다른 타이머는 유지한다.
- 출력 소비자: 기존 `report_tables.py`, #35 성능 비교, #39 A/B 기준선, #42 보고서 후속 작업. 세 JSONL의 스키마와 이전 태그는 유지한다.
- 저장소 산출물은 이 디렉터리 안의 계획·결과·증거 사본이다. implement에서 추적을 위해 필요한 `.gitignore` 예외 한 줄만 추가한다. 이슈 상태는 매니저가, 구현·검증·머지는 해당 워크플로 단계가 소유한다.

## 실행 접근 — implement 승인 이후

1. 읽기 전용 사전 점검으로 4 노드의 코드 동등성·dirty 상태와 골든룰을 기록한다. head running/waiting 모두 0, MemAvailable ≥ 5 GiB를 요구한다. 요청이 남아 있으면 2분 간격으로 최대 3회 확인하며 충족하지 못하면 시작하지 않는다. 기존 prod-final 태그와 로그 유무 및 JSONL 시작 행수를 재확인한다. 충돌 시 기존 자료를 덮거나 지우지 않고 중단 사유를 기록한다.
2. `systemctl --user list-timers`를 로그 첫머리에 남기고, 시각·메모리·SM clock을 `results/prod-final.prestate`에 기록한다. watchdog timer를 stop하고 inactive 상태를 별도로 확인한다(`is-active`의 inactive 종료 코드는 예상 결과). 정지 시각을 기록한다.
3. `B=http://127.0.0.1:8888`, `T=prod-final`, 작업 위치 `~/sglang-cmp`로 지정 레시피의 명령·인자를 그대로 실행한다. suite 전체를 호출하지 않고 위 11개 호출, S128r2, S128r3, `bash ~/dsv41-prep/i34/mixlong_i34.sh prod-final prod-final 8888` 순으로 각각 하나의 SSH 전경 명령을 끝까지 기다린다. 백그라운드 잡 체인은 만들지 않는다. MIXLONG의 기존 내부 병행만 유지하고 `MIX DONE` 이후 두 결과를 확인한다.
4. 각 호출 앞 stamp(날짜·시각·head MemAvailable·SM clock), stdout/stderr, 실제 종료 코드를 `results/prod-final.log`에 append한다. 출력 복제로 종료 코드를 잃지 않게 한다. 각 호출 뒤 5초 쉬고 이번에 추가된 행의 오류, decode의 `streams_ok == c`, health 200, ATTENTION 부재를 확인한다. S128/HOL/longctx/S128r2/S128r3/MIXLONG 직전에는 MemAvailable ≥ 3.0 GiB를 요구한다.
5. 오류·비정상 종료·health 실패·ATTENTION 생성·메모리 가드 실패 시 다음 셀로 진행하지 않고 watchdog timer를 start한 뒤 상태를 기록한다. 서버를 직접 복구하지 않는다. 레시피의 재시도 상한은 셀당 1회이며 `-retry`로 분리한다. 자동 재시도로 실패를 가리지 않고 재개 판단은 매니저에게 남긴다. SSH 단절 등 예외에서도 연결 가능해지는 즉시 타이머 재개와 active 확인을 우선한다.
6. 정상 종료 때도 watchdog timer start/active와 전체 골든룰, 끝 메모리를 기록한다. 정지·재개 시각, 벤치 시작부터 마지막 종료까지의 총시간, kvfs-gc 발화 겹침을 증거로 남긴다.

## 결과 정리

`results.md`에 다음을 담고 `prod-final.log`, prestate와 집계에 필요한 원본 증거를 같은 notes 디렉터리에 복사한다. 기존 원격 파일은 유지한다.

- `report_tables.py results sglang-tp4:SGLang t2w-best:T2W vllm-prod:fork0914 full-combo:combo0915 prod-final:prod-final` 출력을 그대로 수록한다. 5개 비교군이며 디코드는 기존 형식상 군마다 [1]/[3] 두 하위 열이다. 표마다 combo0915 대비 변화와 SGLang 대비 비율을 짧게 설명한다.
- #35의 SGLang=100 항목(산문 c1, 코드 c1, 코드 c4 합산, S128 콜드 프리필, 에이전트 총시간)을 추가한다. 처리량은 prod-final/SGLang×100, 시간은 SGLang/prod-final×100으로 방향을 명시한다. 에이전트 반복값과 평균 집계 기준을 함께 표시한다.
- S128/r2/r3 처리량 원값·최소·최대·평균, 평균 대비 각 회차 편차와 범위, 해당 head 메모리 최저를 기록한다.
- MIXLONG 총시간은 MX4 시작부터 MX4/MXL 중 늦은 종료까지로 계산한다(두 wall의 단순 최댓값이 아님). 기존 기록의 시각·로그를 대조하여 60초 지연 주입을 반영하고 i34-sr4096b 265.4 s와 비교한다. HOL short_wall_s는 7.01/8.71 s와 비교한다.
- 전체 관측 메모리 최저와 시작/끝 값, 총시간, 타이머 정지/재개, 재시도·이상치·GC 겹침을 명시한다. 기존 casebench 샘플과 stamp에서 구한 최저는 관측값이며, 샘플이 없는 구간까지 연속 측정한 최저라고 주장하지 않는다.

## 검증 — validate 단계

- 이번 append 구간과 태그를 함께 확인한다. casebench config=prod-final의 정확한 13태그는 warm/warm2/S8/S32/S128/M3/AG4r1/AG4r2/HOL/S128r2/S128r3/MX4/MXL이다. 중복·누락 없이 각 1행이어야 한다.
- decode tag=prod-final은 prose/code × 1/2/4/8의 8조합, longctx는 1행이다. 스키마에 있는 errors는 모두 비어 있어야 하며, decode streams_ok와 longctx cold/followup 정답·요청 성공도 확인한다. 없는 errors 필드를 임의로 성공 증거로 삼지 않는다.
- 5개 비교군 표를 원본 JSONL과 대조하고 비율·S128 통계·MIXLONG 시간·메모리 최저를 재계산한다. 기존 report 도구의 최초 일치 규칙으로 재시도가 잘못 반영되지 않았는지 확인한다. 실패·재시도 원본은 보존하고 정규 회차 성공으로 둔갑시키지 않는다.
- 실행 뒤 4 노드 골든룰과 watchdog active를 증거로 확인한다. `git diff --stat main`과 파일 목록으로 notes 및 허용된 .gitignore 한 줄 외 변경 0, 코드 변경 0, 삭제 0을 확인한다. 코드 수정이 없으므로 빌드·pytest·torch·새 모델 평가를 실행하지 않는다. 이번 벤치 자체의 성공·긴 세션 정답을 검증한다.

## 범위 밖 및 승인 검토 항목

- #39 A/B, #42 보고서 v5·README 연대기·노드 정리, 서버 재기동·배포·캐시 삭제·sysctl·클럭 조작은 하지 않는다. 화면 변경도 없다.
- **실행기 지침 충돌:** 저장소 AGENTS는 시스템 python3를 금지하지만 지정 코멘트는 비교 조건 유지를 위해 시스템 python3(표준 라이브러리만)를 명시하며 MIXLONG에도 내장되어 있다. 본 계획은 이번 원격 벤치·표 생성에 한해 구체적인 #46 레시피를 적용한다. plan-approve에서 이 예외를 함께 검토한다. 로컬 Python 실행이나 환경 설치는 필요 없다.
- 셀 수는 위 실파일·완료 행수에 근거해 14회 호출로 해석한다. 승인 뒤에도 계획과 다른 부하를 임의 추가하지 않는다.

계획 단계에서는 읽기 전용 조사와 이 파일 작성만 수행했다. 승인 후 implement로 진행하며 이번 제출에서 벤치 실행·커밋·머지·이슈 상태 변경은 하지 않는다.
