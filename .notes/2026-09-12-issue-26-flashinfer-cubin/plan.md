# 이슈 #26 — FlashInfer 버전 불일치 진단 및 수정 계획

## 결론

4대 venv의 `flashinfer-python 0.7.0`과 `flashinfer-cubin 0.6.18` 불일치를 직접 확인했다. 설치된 FlashInfer의 버전 검사 코드가 이 조합을 거부한다. 그러나 **요구한 cubin 0.7.0은 현재 설치할 수 없다**. 4대 모두 지정 버전 dry-run이 실패하고, 공식 PyPI 버전 API도 404다. 따라서 매니저 코멘트 `icmt-908128a6` 3절/5절의 **실패 사유 기록 + README 임시 우회 안내** 대안을 권고한다. 버전 정합 성공으로 보고하지 않는다.

이 문서는 plan 산출물이다. 코드·README·venv 수정, 서버 정지, import/pytest, 커밋·머지, 이슈 상태 변경은 하지 않았다. plan-approve에서 아래 운영 결정까지 검토받는다.

## 증상과 근본 원인

이슈 본문은 FLASHINFER 테스트 2건이 import 단계에서 버전 오류로 실패하고 `FLASHINFER_DISABLE_VERSION_CHECK=1`로 통과했다고 보고한다. 직접 읽은 경로는 다음과 같다.

- `tests/v1/kv_connector/unit/offloading_connector/test_worker.py:294`, `:521`에서 backend를 매개변수화한다. `:328`, `:533`에서 `AttentionBackendEnum[backend].get_class()`를 호출한다.
- `vllm/v1/attention/backends/registry.py:65`가 FLASHINFER 구현을 지정하고, `vllm/v1/attention/backends/flashinfer.py:12`가 `flashinfer`를 import한다. 따라서 본문의 “수집 단계”라는 표현과 별개로, 확인된 두 테스트의 backend import 경로는 테스트 본문에도 있다. 실제 pytest 실패 단계는 이번 plan에서 재실행하지 않아 확정하지 않는다.
- 헤드 `/home/nacyot/vllm-dsv41-venv/lib/python3.12/site-packages/flashinfer/_build_meta.py:2`는 `0.7.0`, 같은 site-packages의 `flashinfer_cubin/_build_meta.py:2`는 `0.6.18`이다.
- 같은 site-packages의 `flashinfer/jit/env.py:88`부터 cubin을 찾고, `:95`의 조건은 우회 환경변수가 없으며 Python 버전이 unknown이 아니고 두 버전이 다르면 `:100`에서 RuntimeError를 발생시킨다. 설치된 cubin이 존재하므로 “패키지가 없을 때 기본 캐시 디렉터리를 쓰는” `:109` 분기로 구제되지 않는다.
- `flashinfer_python-0.7.0.dist-info/direct_url.json:1`은 로컬 소스 `file:///home/nacyot/dsv41-prep/flashinfer-src`에서 설치되었음을 보여준다. `METADATA:15` 이후 의존성 목록에는 cubin 동버전 요구가 없다. Python 패키지의 소스 빌드와 별도 cubin 배포본이 서로 다른 버전으로 공존하며, 설치 시가 아닌 import 시 불일치가 드러나는 환경 문제다. 누가 언제 이 조합을 만들었는지는 확인하지 않았다. 구 소스 사본을 실행하거나 재설치에 사용하지 않는다.
- **서빙 정상의 이유도 확인했다.** 로컬 및 헤드의 `deploy/gb10-cluster/dsv41/serve-node.sh:39`가 `FLASHINFER_DISABLE_VERSION_CHECK=1`을 export한다. “서빙에서는 이 코드를 쓰지 않을 것”이라는 가정이 필요 없다. health 200은 버전 정합의 증거가 아니다.

추가로 설치 중 지연 로드 위험은 실제 코드에 있다. 헤드 site-packages의 `flashinfer/jit/cubin_loader.py:314` 콜백이 artifact를 요청하며, `:216`은 로컬 파일을 읽고 `:190`은 파일을 연다. 실패 시 `:221` 이후 다운로드 금지 여부에 따라 예외 또는 다운로드로 이어진다. 온라인 패키지 교체가 안전하다고 간주할 수 없으므로 venv 변경은 반드시 정지 창 안에서만 한다.

## 읽기 전용 실측

2026-09-12 KST, plan 단계에서 각 노드를 순서대로 SSH 조회했다. Python/torch/FlashInfer import 없이 파일을 읽고 uv resolver만 실행했다.

| 노드 | Python / cubin dist-info 버전 | cubin 0.7.0 dry-run | 서버 / cap / SM |
| --- | --- | --- | --- |
| gx10-6040 | 0.7.0 / 0.6.18 | 해당 버전 없음 | active / active / 1989 MHz |
| gx10-f323 | 0.7.0 / 0.6.18 | 해당 버전 없음, exit 1 | active / active / 1989 MHz |
| gx10-37cc | 0.7.0 / 0.6.18 | 해당 버전 없음, exit 1 | active / active / 1989 MHz |
| gx10-27c4 | 0.7.0 / 0.6.18 | 해당 버전 없음, exit 1 | active / active / 1989 MHz |

명령은 각 노드에서 다음과 같다. `uv pip install --help`로 옵션을 먼저 확인했다.

```bash
~/.local/bin/uv pip install --dry-run \
  --python ~/vllm-dsv41-venv/bin/python flashinfer-cubin==0.7.0
```

공통 진단: `Because there is no version of flashinfer-cubin==0.7.0 ... requirements are unsatisfiable.` 헤드 명령은 뒤에 메모리 조회가 붙어 셸 전체 종료값이 0이므로 설치 성공으로 해석하지 않는다.

[공식 PyPI 0.7.0 API](https://pypi.org/pypi/flashinfer-cubin/0.7.0/json)는 웹 조회와 헤드 curl 모두 HTTP 404였다. 현재 지정 배포 경로에서 제공되지 않는다는 의미이며, 모든 사설/비공개 배포물의 부재까지 주장하지 않는다.

`/opt/homebrew/bin/bash deploy/gb10-cluster/dsv41/dsv41_ctl.sh status` 결과 head MemAvailable 4.62 GiB, 나머지 9.32/9.10/9.29 GiB, :8889 health 200. 헤드 로그의 마지막 loggers 행은 Running 0 / Waiting 0이다. earlyoom 로그 tail은 메모리 보고뿐이었으나 전체 기간 kill 0 검증을 대신하지 않는다. 실제 창 직전에 상태·프로브 프로세스를 다시 확인해야 한다.

## 수정 방법과 단계 배치

1. **implement:** `deploy/gb10-cluster/dsv41/README.md`의 Local selftest 인근에 노드 단위 테스트 안내를 추가한다. 서버가 내려간 창에서만 실행, 두 패키지 동버전이 원칙, 불일치 시 명령 단위 `FLASHINFER_DISABLE_VERSION_CHECK=1`이 임시 우회임을 명시한다. 우회는 패키지 호환성이나 버전 정합을 보장하지 않는다. 영구 shell 설정이나 서버 설정은 추가하지 않는다.
2. 현재 0.7.0 부재와 4대 dry-run 실패를 `.notes/2026-09-12-issue-26-flashinfer-cubin/results.md`에 기록한다. 실패가 이미 resolver 단계에서 확인됐으므로 같은 설치를 실제 실행해 서버 정지를 소비하지 않는 대안을 권고한다.
3. **validate:** 승인된 아래 분기에 따라 문서 검토 및 운영 검증을 수행하고, 실행/미실행/실패를 구분해 results에 남긴다. **merge:** 해당 단계에서만 README 변경을 병합한다. 노드 코드 반영은 README만 바뀌므로 불필요하다. 이슈 in_progress/done은 매니저가 처리한다.

## plan-approve에서 필요한 운영 결정

**권고: 설치 불가 대안으로 문서만 수정하고 정지 창을 열지 않는다.** 코멘트는 실패 시 문서 대안을 허용하지만 완료 기준에는 head import/pytest 결과도 요구한다. 새 import·pytest를 생략하고 이번 읽기 전용 증거 및 기존 이슈의 우회 통과 기록으로 대안 완료를 인정할지 승인받는다. 이 경우 import 무우회 통과/pytest 재검증/재기동 후 8K는 미실행으로 명시하고, validate에서 상태·메모리·cap·earlyoom을 확인한다. “창 닫음” 기록은 실제로 열지 않았다는 뜻을 분명히 적어 매니저가 #8 dispatch를 판단할 수 있게 한다.

**재검증 창을 요구하는 경우:** 한 번의 정지→검증→복구 창으로만 진행한다. 이때 추가 제약이 있다. `dsv41_ctl.sh:120` start 및 `:142` stop에 `/dev/shm` 글롭 `rm`이 있어 사용자 금지와 충돌한다. 원본 stop/start를 그대로 실행하지 않는다. README 외 tracked 코드 범위는 유지하고, `mktemp -d`에 삭제 구문을 제외한 운영용 임시 사본을 만들어 diff 검토 후 사용하는 방식까지 이번 게이트에서 승인받는다. 필수 잔재 정리가 발견되면 실경로·소유 프로세스·허용 범위를 확인한 명시적 파일 목록만 다루며, 공유 디렉터리 전체나 kvfs는 삭제하지 않는다. 경로를 입증할 수 없으면 중단한다.

창 직전 이슈 최신 코멘트/충돌 규칙, health 200, Running/Waiting 0 및 `kvoff_probe` 프로세스 부재를 확인한다. #28의 스크립트 심링크 작업과 충돌하지 않도록 노드 스크립트를 덮어쓰지 않는다. 4대 serve/frontend unit 및 관련 서버 프로세스 종료를 확인한 뒤에만 아래 검증을 head에서 수행한다.

```bash
cd ~/vllm-dsv41
env -u FLASHINFER_DISABLE_VERSION_CHECK \
  ~/vllm-dsv41-venv/bin/python -c 'import flashinfer; print(flashinfer.__version__)'
FLASHINFER_DISABLE_VERSION_CHECK=1 ~/vllm-dsv41-venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/offloading_connector/test_worker.py \
  -k 'register_kv_caches and FLASHINFER' -v
```

현재 조합에서는 무우회 import 실패가 예상되므로 수정 성공으로 취급하지 않는다.
현재 조합을 유지한다면 pytest는 한 번만 임시 우회로 실행하고 2건 선택 여부와 결과를 기록한다.

새 공식 0.7.0 배포물이 제공되어 승인 계획이 정합 분기로 변경될 때만, 창 전에 4대 dry-run으로 cubin 외 변경이 없는지 확인한다. 창 안에서 노드별 설치 후 dist-info/실버전을 재확인하고 위 import와 pytest를 **모두 우회 없이** 실행한다. 설치 실패 시 원상 유지 여부를 실측하며 자동으로 0.6.18이 보존됐다고 가정하지 않는다. 다운그레이드·소스 빌드·버전 문자열 위조는 하지 않는다.

재기동은 환경 override/fixture/캡 우회를 제거한 채 채택 `dsv41.env` 기본 TP=4로 한다. 모든 작업은 작은 포그라운드 단계로 수행하고 긴 기동은 30–60초마다 관찰한다. health 200 뒤 head 최신 repo의 `.venv/bin/python`으로 `deploy/gb10-cluster/dsv41/kvoff_probe.py i26-smoke S26 700`을 한 번 실행해 정답을 확인한다. 이 스크립트의 import는 stdlib만임을 읽어 확인했으며 torch 프로세스 금지를 우회하지 않는다. 재기동 구간의 flashinfer 경고와 earlyoom kill을 기록한다. 종료 기준은 health 200, 4대 cap active·SM 1989 MHz, head MemAvailable ≥4.5 GiB, 작업 구간 earlyoom kill 0이다. 실패 시 성공으로 닫거나 임의의 두 번째 창을 열지 않고 복구 상태를 보고한다.

## 검증 및 제외 범위

README diff와 명령·환경변수 설명을 직접 검토하고 `git diff --check`를 실행한다. 문서 변경만을 위한 신규 테스트나 GPU 벤치마크는 만들지 않는다. `vllm/`, 테스트 코드, dsv41.env, tracked 제어 스크립트, 다른 패키지 및 kvfs는 수정하지 않는다. 새 소스 빌드나 광범위한 모델 평가, clock/persistence 조작은 범위 밖이다. 메인 워커가 수행하며 서브에이전트/gptcli 작업은 필요하지 않다.
