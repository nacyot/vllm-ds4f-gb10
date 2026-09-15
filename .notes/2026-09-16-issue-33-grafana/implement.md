# 이슈 #33 구현 기록

작성 2026-09-16 KST. 계획은 `plan.md` 에 있다. 프로덕션 :8888 에는 요청을 보내지 않았고 설정 변경도 없었다.

## homelab (Forgejo `main`)

homelab 워크트리 `~/workspace/worktrees/nacyot/homelab/issue-33-dsv41-panels-20260916` 에서 작업했다(브랜치 `issue-33-dsv41-panels`, base `origin/main` `933f3ad`). 로컬 `main` 과 그 미커밋 편집은 건드리지 않았다. push 직전 `origin/main` 에 `91317fa`(lainyzine CronJob, 대상 파일과 무관)가 올라와 있어 그 위로 rebase 했다. `git push origin HEAD:main` 은 fast-forward(`91317fa..315f681`) 로 들어갔다.

| 커밋 | 내용 |
| --- | --- |
| `53e32a5` | `grafana: count computed prefill tokens and drop DS4F-only panels on ds4f-vllm (#33)` |
| `315f681` | `victoriametrics: drop the dspark detector rules from ds4f.yml (#33)` |

대시보드 JSON 은 jq 필터로 바꿨다(`jq --indent 1`, 왕복 결과 원본 포맷과 동일).

- 패널 수 61, id 중복 0, gridPos 겹침 0, 제목 `DSv4.1F vLLM (gx10)`.
- diff 의 추가 줄은 계획 2-1 표의 항목뿐이다(62/63/68 삭제가 삭제 줄 대부분).
- 바뀐 expr 9 개(43, 42 A·B, 3, 72, 12, 17, 18, 40)를 VM `/api/v1/query` 에 보냈다(`$__rate_interval` → `1m`). 모두 `status: success` 였다. 72 는 최근 5 분에 끝난 요청이 없어 결과가 비었다(0/0).

## CT116 vmalert 규칙

2026-09-16 01:33~01:35 KST, `ssh ser9t 'pct exec 116 -- …'`.

1. 교체 전 라이브 `/opt/victoriametrics/rules/ds4f.yml` 이 레포 원본(`2ae790c`) 과 같음을 확인했다(diff 0).
2. dryRun: 새 파일을 CT116 `mktemp -d`(`/tmp/tmp.85XxJCQpB7`) 에 두고 실행 중인 vmalert 이미지 `c6e6c1ef6e43` 으로 `-rule=/r/ds4f.yml -dryRun` 을 돌렸다. 로그는 1 파일 읽음이고 오류는 없었다. alert 는 5 개다.
3. `cp -p ds4f.yml ds4f.yml.bak-20260916-dsv41`(4689 B, 새 경로가 없음을 먼저 확인) 뒤 새 파일(3271 B) 로 교체했다. sha256 앞 16 자리는 원본 `9ac609f863854cd9`, 새 파일 `7d7077efc3695d49`(레포 커밋본과 같음) 이다. vmalert 가 `-rule=/rules/*.yml` 로 읽으므로 `.bak-…` 파일은 로드되지 않는다.
4. `docker kill --signal=HUP vmalert` 로 재적재한 뒤 `/api/v1/rules` 를 확인했다. 그룹 41 개, lastError 0 이다. `ds4f.agent-stall` 은 3 규칙(`DS4FGreedyStallFingerprint`, `DS4FRunawayFinish`, `DS4FLongDecode`) 이 모두 health ok 이고 01:34:51 KST 에 평가됐다. `ds4f.memory` 는 2 규칙이다.
5. `DS4FRunawayFinish` 는 재적재 직후 firing 이었다. 최근 10 분 `finished_reason="length"` 종료가 실제 트래픽에서 2 건 있었기 때문이고, 규칙은 바꾸지 않았다.

다른 규칙 파일은 건드리지 않았다. 삭제한 파일은 없다(임시 디렉터리도 그대로 둠). 안전 확인창은 뜨지 않았다.

## Flux·Grafana 배포

- `ssh ser9 kubectl -n flux-system get kustomization`: `apps`, `infra`, `flux-system` 이 `main@sha1:315f681` 을 적용했고 Ready 다. `immich` 는 push 전부터 `dependency 'flux-system/infra' is not ready` 로 False 였고 이번 변경과 무관하다.
- ConfigMap `grafana/grafana-dashboards` 의 `ds4f-vllm.json` 제목은 새 값이다.
- 라이브 `/api/dashboards/uid/ds4f-vllm` 는 `meta.updated` 2026-09-16 01:35:24 KST 에 바뀌었다(push 뒤 약 2 분). 제목 `DSv4.1F vLLM (gx10)`, 패널 61, `provisioned: true` 다. 패널별 id·title·gridPos·description·expr 이 커밋본과 모두 같다(`jq -S` diff 0).

## 이 레포

- `deploy/gb10-cluster/dsv41/README.md` 에 `### Metrics in Grafana (issue #33)` 한 문단을 넣었다(`e0f1e74ed9`).
- 코드 변경은 없다.
