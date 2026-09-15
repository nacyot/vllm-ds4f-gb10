# 이슈 #35 검증 기록 (2026-09-16)

계획: `plan.md`. 범위 정본: #35 매니저 레시피 코멘트 `icmt-e4fa9f48`.
**엔진 코드 변경 0, 서버 재기동 0, 벤치 0.** `--force` 푸시·`git reset --hard`·히스토리 재작성 없음.

## 1. main push

푸시 전(2026-09-15T20:37:21Z UTC = 09-16 05:37 KST):

| 대상 | sha |
| --- | --- |
| 로컬 `main` | `caa64758296a4eb4d388d5c19308328719ed56fb` |
| GitHub `github` `dsv41-gb10` | `dccd5b80f9bff19f14232ff60b8b5b221944a2d8` |
| Forgejo `origin` `main` | `dccd5b80f9bff19f14232ff60b8b5b221944a2d8` |

실행과 출력:

```text
$ git push github main:dsv41-gb10
To github.com:nacyot/vllm-ds4f-gb10.git
   dccd5b80f9..caa6475829  main -> dsv41-gb10

$ git push origin main:main
To git.tail39057.ts.net:nacyot/vllm-ds4f-gb10.git
   dccd5b80f9..caa6475829  main -> main
```

두 출력 모두 `..`(fast-forward) 이고 `+`(강제) 가 아니다. 푸시 뒤:

| 대상 | sha |
| --- | --- |
| GitHub `dsv41-gb10` | `caa64758296a4eb4d388d5c19308328719ed56fb` |
| Forgejo `main` | `caa64758296a4eb4d388d5c19308328719ed56fb` |
| GitHub `gb10-longctx-offload`(V4.0, 안 건드림) | `ab7723d3ff89465a49013e606c654b13d2c7551a` |

49 커밋이 올라갔다. README 내용은 손대지 않았다.

## 2. 노드 sync (4 대, 노드별 한 커밋)

절차는 계획 2-2 그대로다. 워커 3 대는 `fork` 리모트(`https://github.com/nacyot/vllm-ds4f-gb10.git`) 를 새로 추가했고 head 는 이미 있었다. 네 대 모두 `git fetch fork dsv41-gb10` 의 `FETCH_HEAD` 가 `caa64758296a4eb4d388d5c19308328719ed56fb` 였다.

| 노드 | sync 전 HEAD | sync 커밋 | 게이트 경로 수 | 커밋 뒤 `git diff --stat HEAD FETCH_HEAD -- . ':!.notes'` | `git status --short` |
| --- | --- | --- | ---: | --- | ---: |
| gx10-6040 (head) | `88744c769e` | `dedf44fec46c8445ca4470e7e215b0823e76d600` | 8 | 빈 출력 | 0 줄 |
| gx10-f323 | `d77882fc7` | `945767da6ea7b7ddd5dc807c067515ce170725f6` | 20 | 빈 출력 | 0 줄 |
| gx10-37cc | `3d81648af` | `26eab528cd8ae7caa691da82cea6c3bf64d9fbd0` | 20 | 빈 출력 | 0 줄 |
| gx10-27c4 | `fe5e0aff1` | `22d5358ef658d2f05e328a66e8aa939d5bd1d865` | 20 | 빈 출력 | 0 줄 |

게이트(`git diff --stat HEAD FETCH_HEAD -- . ':!.notes'`) 는 네 대 모두 계획 0 절에서 예상한 목록과 정확히 일치했다. 목록 밖 파일은 없었다 — 엔진 트리는 갈라지지 않았다.

head 8 경로(911 insertions, 115 deletions):

```text
 .gitignore                                 |   5 +
 README.ko.md                               | 117 +++++++++
 README.md                                  | 171 ++++++-------
 deploy/gb10-cluster/dsv41/README.md        | 376 ++++++++++++++++++++++++++---
 deploy/gb10-cluster/dsv41/clock_ctl.sh     |  48 ++++
 deploy/gb10-cluster/dsv41/clock_summary.py |  61 +++++
 deploy/gb10-cluster/dsv41/test_headroom.py | 246 +++++++++++++++++++
 vllm/v1/kv_offload/tiering/manager.py      |   2 +-
```

워커 3 대는 여기에 벤치·프로브 스크립트 갱신(`bench.py bench2.py decode_gap_probe.py divergence.py kvoff_concurrent.py kvoff_probe.py memlog.py prefill_probe.py smoke.py`) 과 신규 `kvfs_verify.py test_kvfs_verify.py test_memlog.py` 가 더해져 20 경로(2,098 insertions, 199 deletions) 였다. 전부 head 에서만 쓰는 스크립트라 실사용 경로에 없다.

`.notes/` 는 동기화하지 않았다. 네 노드 모두 `2026-09-12-issue-24-frontend-split/plan.md` 하나가 그대로 남아 있다.

### 계획과 달랐던 점 하나

워커 3 대에 git author identity 가 없어 첫 `git commit` 이 `Author identity unknown` 으로 거절됐다. 파일은 이미 스테이징된 상태(20 경로)였고, **저장소 로컬**(`--global` 아님)로 `user.name=nacyot` / `user.email=propellerheaven@gmail.com` 을 설정해 커밋했다. head 의 기존 설정·커밋 author 와 같은 값이다. head 는 이미 설정돼 있어 건드리지 않았다.

## 3. 서비스가 바뀌지 않았다는 근거

- 유닛이 실제로 읽는 `dsv41_unit.sh dsv41_lib.sh dsv41.env dsv41_ctl.sh serve-node.sh dsv41_warmup.py dsv41_watchdog.sh kvfs_gc.sh` 와 `systemd/` 하위는 main 과 블롭까지 동일해, 게이트 diff 에 아예 나타나지 않았다(위 목록 확인).
- `vllm/v1/kv_offload/tiering/manager.py` 는 `@override` 위치 2 줄만 바뀌었고 이미 import 된 모듈이라 실행 중 프로세스에 반영되지 않는다. 다음 재기동 때 main 과 같은 코드가 올라온다.
- `dsv41-serve` 기동 시각은 sync 전후로 동일하다(아래). sync 시각은 09-16 05:37~05:38 KST 다.

| 노드 | `dsv41-serve` ActiveEnterTimestamp |
| --- | --- |
| gx10-6040 | Wed 2026-09-16 04:30:04 KST |
| gx10-f323 | Wed 2026-09-16 04:29:57 KST |
| gx10-37cc | Wed 2026-09-16 04:29:42 KST |
| gx10-27c4 | Wed 2026-09-16 04:29:27 KST |

## 4. 골든룰 (sync 직후)

| 노드 | dsv41-serve | Environment | override.env | gpu-clock-cap | SM 클럭 | `vllm-dsv41` dirty |
| --- | --- | --- | --- | --- | --- | --- |
| gx10-6040 | active | 비어 있음 | 0 bytes | active | 1989 MHz | 0 줄 |
| gx10-f323 | active | 비어 있음 | 0 bytes | active | 1989 MHz | 0 줄 |
| gx10-37cc | active | 비어 있음 | 0 bytes | active | 1989 MHz | 0 줄 |
| gx10-27c4 | active | 비어 있음 | 0 bytes | active | 1989 MHz | 0 줄 |

`gpu-clock-cap.service` 는 네 대 모두 `ExecStart=/usr/bin/nvidia-smi -lgc 300,2000` 그대로다. `nvidia-smi` 클럭 명령은 실행하지 않았다.

head 추가: `dsv41-watchdog.timer` active, `http://127.0.0.1:8888/health` 200, `/v1/models` 가 `deepseek-v4.1-flash`(`max_model_len` 524288) 응답, MemAvailable 9.90 GiB, 워치독 state 에 ATTENTION 없음.

## 5. 저장소 변경

코드 변경 0. 이 워크트리에서 건드린 파일은 `.gitignore`(노트 디렉터리 negation 한 줄, 기존 215~218 행 옆)와 `.notes/2026-09-16-issue-35-epic-wrapup/` 뿐이다.

## 6. 이번에 하지 않은 것

- 머지 뒤 재푸시. 이 노트 커밋은 push 뒤에 main 으로 들어가므로 두 원격은 노트 커밋만큼 뒤진 상태로 끝난다 — 레시피가 허용한 상태이고 다음 push 때 올라간다.
- 이슈 본문 정정과 이슈 닫기(매니저 몫), 메모리 파일 갱신(매니저 몫).
- 벤치·프로파일 재실행, 서버 재기동, `dsv41-override.env` 편집, 파이썬 재설치.

## 7. 오너 결정 대기 (되묻지 않고 인계)

- 옛 저장소 `~/dsv41-prep/kvfs` 37 GiB(4 노드).
- f323 도커 이미지 `dsv41-4x-spark:local` 33.2 GB 와 dangling `<none>` `171c04994a8d`(레이어 공유).
- #45 워커 NFS `/mnt/kvdisk` fstab — 부팅 복원이 안 된다.
