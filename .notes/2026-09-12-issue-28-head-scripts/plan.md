# 이슈 #28 계획: dsv41-prep 중복 스크립트 정리

근거: 이슈 #28 본문과 매니저 레시피 `icmt-0458006a`, 현재 dsv41 README 및 스크립트. 이 단계는 계획서 작성만 수행한다. 구현·검증·머지는 승인 후 각 워크플로 단계가, 이슈 상태는 매니저가 소유한다.

## 현재 동작과 문제

`~/vllm-dsv41/deploy/gb10-cluster/dsv41/`가 정본이지만, 노드의 `~/dsv41-prep`에 독립 사본이 남아 있다. 매니저 조사에서 head의 중복 Python 파일은 10개이며 worker 3대에도 후보 파일이 있다. 구 `kvoff_concurrent.py`는 콤마 salt를 처리하지 못하고, 구 `kvoff_probe.py`·`prefill_probe.py`에는 #19 메모리 가드가 없다. 내용이 같은 사본도 다음 반영 때 다시 갈라진다.

README는 이미 레포 경로 실행과 구 사본 사용 금지를 명시한다. 따라서 실행 로직을 새로 만들지 않고 기존 경로가 정본을 참조하도록 바꾼다. `kvoff_concurrent.py`와 `prefill_probe.py`는 `kvoff_probe`를 import하며, 후자는 자기 파일 디렉터리를 import 경로에 추가한다. `bench2.py`는 `abspath(__file__)` 옆의 `bench_prompts_v1.json`을 읽으므로 심링크 실행 시 인접 데이터 확인도 필요하다. 기존 출력 기본 경로인 `~/dsv41-prep/bench`는 유지한다.

head에서 읽기 전용 확인한 `bench2_summary.py`는 레포에 없는 43줄짜리 실험 도구다. C4가 없을 때 빈 집계가 실패하며 태그 기반 인자와 비교 출력이 결합되어 있다. 작은 운영 정리 범위를 유지하기 위해 신규 레포 편입·수정은 하지 않고, 레시피 3절의 보관 후 후속 메모 방안을 선택한다.

## 변경과 접근

1. 구현 직전 이슈 코멘트를 다시 확인한다. gx10-6040, gx10-f323, gx10-37cc, gx10-27c4를 한 대씩 읽어 실제 홈·레포·보관 경로, Git 상태, `.py`/`.sh` 항목의 종류와 정본 존재 여부, `cmp` 결과를 확정한다. 조사 표와 이동 전후 `ls -la`는 같은 디렉터리의 `results.md`에 기록한다. head 예상 목록은 `kvoff_probe.py`, `kvoff_concurrent.py`, `prefill_probe.py`, `bench.py`, `prof_summary.py`, `smoke.py`, `bench2.py`, `divergence.py`, `memlog.py`, `memtrace_summary.py`이며 실제 노드별 목록이 기준이다.
2. 노드별 `~/dsv41-prep/superseded-2026-09-12/`에 중복 일반 파일만 이름을 하나씩 명시해 이동한다. 대상 디렉터리의 실경로와 파일 충돌을 먼저 확인하고 기존 보관물을 덮어쓰지 않는다. 이미 올바른 심링크인 항목은 유지하며 잘못된 링크나 예상 밖 파일은 별도 기록 후 재판단한다. 이동 직전 조사 당시와 파일이 동일한지도 확인한다.
3. 원래 이름에 해당 노드의 `/home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/<name>`을 가리키는 심링크를 만든다. 정본이 없거나 경로가 예상 범위를 벗어나면 이동하지 않는다. 각 파일의 보관과 링크 생성 결과를 확인한 다음 파일로 진행한다. 실패 시 새 링크를 별도 임시 위치로 옮기고 보관 원본을 복구하는 방식으로 삭제 없이 되돌린다.
4. head의 `bench2_summary.py`는 동일 보관 디렉터리로 명시 이동하고 대체 링크는 만들지 않는다. 기존 태그 요약 명령은 더 이상 제공되지 않음을 결과와 후속 메모에 명시한다. 나머지 레포에 없는 실험 파일, JSON, 로그, 벤치 결과, kvfs 데이터는 보존한다.
5. README `Operating rules`의 기존 레포 실행 지침에 다음 한 문장을 추가한다: “Matching script entries in `~/dsv41-prep/` are symlinks to `~/vllm-dsv41/deploy/gb10-cluster/dsv41/` since 2026-09-12 (#28).” 모든 prep 파일이 링크라는 오해를 피한다.

## 불변 조건과 영향 사이트

- 네 노드의 기존 스크립트 경로를 사용하는 운영자·워커가 최신 정본과 메모리 가드를 사용하게 된다. 다음 `git am` 반영도 링크에 자동 반영된다. 레포 직접 실행 경로와 인자·출력 계약은 유지한다.
- `bench2.py` 인접 JSON은 삭제하거나 교체하지 않고 정본과의 일치 여부를 확인한다. 누락·불일치로 링크 경유 동작이 달라지면 해당 항목을 완료로 판정하지 않고 범위 재검토 사유를 기록한다. Python import 실패도 추측으로 넘기지 않는다. 레시피의 exec 래퍼 대안은 완료 기준인 전부 심링크와 다르므로 필요성이 입증되면 승인 범위를 다시 검토한다.
- #26의 venv 정합 작업 및 개발 큐와 겹치는 venv·서버·코드 변경은 하지 않는다. 노드 코드 반영이 필요한 단계에서는 `git format-patch | git am`만 사용하고, 적용 중인 다른 패치나 dirty 상태를 덮어쓰지 않는다.
- TP=4와 채택 `dsv41.env` 구성을 유지한다. `dsv41_ctl.sh stop/start`, 클록 변경 명령, cap 서비스 stop/restart, kvfs 삭제, venv 변경은 실행하지 않는다. 서버가 살아 있는 노드에서는 pytest와 torch import 프로세스를 실행하지 않는다.
- 작업은 한 단계씩 포그라운드로 수행하며 실행 중에는 30~60초 간격으로 관찰한다. 임시 산출물은 `mktemp -d`에 둔다. `rm`은 쓰지 않는다. 안전 확인창이 발생하면 실제 대상·효과 확인 후 해당 질문만 처리하고 시각·머신·명령 요약·판단·재개 결과를 비밀값 없이 기록한다.

## 검증

1. 변경 전후 네 노드의 대상 목록을 비교한다. 레포와 이름이 겹치는 prep 최상위 `.py`/`.sh` 일반 사본이 0개이고 모든 대상 링크의 실경로가 정본인지 확인한다. 보관 원본의 체크섬·파일 수와 이동 목록을 대조하며 다른 실험 파일이 유지됐는지 확인한다.
2. head의 `~/vllm-dsv41/.venv`가 기존 환경을 가리키는지 재확인한다(계획 시 `/home/nacyot/vllm-dsv41-venv` 링크 확인). Python은 이 `.venv/bin/python`을 사용한다. 노드에 새 환경이나 의존성을 설치하지 않는다.
3. head에서 prep 디렉터리 기준으로 `.venv/bin/python`의 절대 경로를 사용해 `prefill_probe.py --help`와 `from kvoff_probe import HeadroomError, metrics, run_prompt`를 확인한다. 정본 코드의 표준 라이브러리 import만 허용한다. `kvoff_probe` import와 help도 `/v1/models` 조회가 발생하므로 완전한 오프라인 검사로 취급하지 않는다. `kvoff_concurrent.py` 자체는 import만 해도 추론 코드가 실행되므로 검사용으로 import하지 않는다.
4. health와 메모리 여유가 정상인 상태에서 head의 prep 경로로 `~/vllm-dsv41/.venv/bin/python kvoff_probe.py i28-smoke S28 700`을 한 번 실행한다. 정상 종료, 응답 정답 여부, 실제 prompt tokens(약 8K), 메모리와 health를 기록한다. 긴 prefill이나 동시 대형 벤치는 하지 않는다.
5. 종료 시 :8889 health 200, 네 `gpu-clock-cap.service` active, `nvidia-smi --query-gpu=clocks.sm --format=csv,noheader` 유휴 관측 1989 MHz, head MemAvailable ≥4.5 GiB, 채택 TP=4 구성을 확인한다. 미달 시 서버 재시작이나 클록 조작으로 복구하지 않고 추가 부하를 중단해 기록한다. 조건이 회복되기 전 완료로 보고하지 않는다.
6. README 및 기록 파일 diff/공백 검사를 수행한다. 실행 로직 변경이 없으므로 신규 단위 테스트나 모델 정확도 평가를 만들지 않으며 위 링크·import·8K 실측으로 이 변경의 동작을 검증한다.

## 범위와 승인

레포 변경은 README 한 문장과 `.notes/2026-09-12-issue-28-head-scripts/{plan.md,results.md}` 작업 기록이다. 노드 변경은 prep의 확정 목록 이동과 링크 생성이다. 화면 변경, vLLM 코드 변경, summary 도구 신규 개발은 없다. 오너만 결정해야 할 추가 질문은 현재 없다. 계획 제출 후 plan-approve에서 사람 승인을 기다리고 implement로 직접 전이하지 않는다.
