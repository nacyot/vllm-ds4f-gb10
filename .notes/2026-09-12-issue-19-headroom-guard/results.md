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
