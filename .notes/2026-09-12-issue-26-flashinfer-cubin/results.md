# 이슈 #26 작업 결과

## 구현 결정

plan-approve가 승인한 계획의 권고안(패키지 부재 시 문서 대안)을 적용했다.
README에 노드 테스트의 정지 창 조건, 두 패키지 버전 정합 원칙,
명령 단위 `FLASHINFER_DISABLE_VERSION_CHECK=1` 임시 우회와 한계를 추가했다.
버전 불일치 자체는 남아 있으며, 버전 정합 완료로 보고하지 않는다.

창 닫음: **정지 창을 열지 않았다(0회)**. 서버 정지·기동, venv 변경,
노드 코드 반영은 없었다. 다음 워커 dispatch 및 이슈 상태는 매니저가 판단한다.

## plan 단계에서 확인한 설치 불가 근거

2026-09-12 KST, 4대 모두 다음 dist-info를 파일 조회로 확인했다.

| 노드 | flashinfer-python | flashinfer-cubin | dry-run 결과 |
| --- | --- | --- | --- |
| gx10-6040 | 0.7.0 | 0.6.18 | 지정 버전 없음 |
| gx10-f323 | 0.7.0 | 0.6.18 | 지정 버전 없음, exit 1 |
| gx10-37cc | 0.7.0 | 0.6.18 | 지정 버전 없음, exit 1 |
| gx10-27c4 | 0.7.0 | 0.6.18 | 지정 버전 없음, exit 1 |

헤드의 `uv pip install --help`로 옵션을 확인한 뒤 각 노드에서 사용했다.

```bash
~/.local/bin/uv pip install --dry-run \
  --python ~/vllm-dsv41-venv/bin/python flashinfer-cubin==0.7.0
```

공통 오류는 `there is no version of flashinfer-cubin==0.7.0` 및
`requirements are unsatisfiable`이다. 헤드 명령 뒤에 메모리 조회가 있어
셸 전체 exit 0이 반환됐지만, resolver는 같은 실패를 출력했다.
실제 설치·삭제·롤백은 실행하지 않았다.

[공식 PyPI 버전 API](https://pypi.org/pypi/flashinfer-cubin/0.7.0/json)는
웹 조회와 헤드 curl에서 모두 HTTP 404였다. 다른 사설 배포물의 존재 여부는
확인하지 않았으며, 다른 버전 설치나 소스 빌드로 범위를 넓히지 않았다.

헤드 site-packages의 `flashinfer/jit/env.py:95`는 환경변수 우회가 없고
두 패키지 버전이 다르면 RuntimeError를 낸다. `serve-node.sh:39`가 이미
우회를 export하므로 서버 health 200과 단독 테스트 import 실패는 양립한다.
정확한 코드 경로와 진단 근거는 같은 디렉터리의 `plan.md`에 있다.

## 검증 구분

- 기존 이슈 본문은 우회 시 FLASHINFER 테스트 2건 통과를 보고한다.
  이번 워커가 재실행한 결과가 아니다.
- head 무우회 import, pytest 2건, 재기동 후 8K 프로브는 미실행이다.
  승인된 문서 대안에 따라 서버를 정지하지 않았으므로 torch import를
  수반하는 테스트를 실행하지 않았다.
- plan 단계 status는 :8889 health 200, 4대 serve/cap active·SM 1989 MHz,
  head MemAvailable 4.62 GiB였다. 이는 구현 후 종료 상태 검증이 아니다.
- 구현 변경의 문서 검사 결과는 아래에 기록하고, 최종 운영 상태 및
  작업 구간 earlyoom 검증은 다음 validate 단계에서 기록한다.

## 구현 단계 문서 검사

기존 workstation `pre-commit`을 사용해 hooks를 설치했다.
다음 명령은 exit 0이며 markdownlint-cli2, typos 및 적용 가능한 hooks가
통과했다. Python/GPU 테스트는 실행하지 않았다.

```bash
pre-commit run --files deploy/gb10-cluster/dsv41/README.md \
  .notes/2026-09-12-issue-26-flashinfer-cubin/plan.md \
  .notes/2026-09-12-issue-26-flashinfer-cubin/results.md
git diff --check
```

## 변경 범위

README와 이슈 작업 메모만 변경했다. vLLM 코드, 테스트 코드, 채택 구성,
노드 venv 및 제어 스크립트는 변경하지 않았다. 삭제 명령이나 안전 확인창,
GPU clock/persistence 조작은 없었다. AI assistance: Codex.
