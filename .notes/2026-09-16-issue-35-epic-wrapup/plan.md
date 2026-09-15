# 이슈 #35 계획: 에픽 마무리 — main push 와 4 노드 문서 sync

작성 2026-09-16, 워크트리 `issue-35-epic-wrapup`, 기준 main `caa6475829`.
범위 정본은 #35 매니저 레시피 코멘트 `icmt-e4fa9f48` 와 인계 코멘트 `icmt-27553bc8` 다. 이 계획은 그 범위를 실행 순서로 옮긴 것이고 범위를 넓히지 않는다.

하위 8 건(#36 #37 #38 #39 #40 #41 #42 #46)이 전부 done 이라 이 에픽에 실행으로 남은 것은 오너 규칙 "main push·노드 sync 는 다 끝난 뒤" 에 해당하는 두 가지뿐이다. **엔진 코드 변경 0, 서버 재기동 0, 벤치 0.**

## 0. 계획 단계에서 실측한 것

방향이 갈릴 수 있는 지점만 확인했고, 결과는 전부 매니저 코멘트와 일치한다.

- 로컬 `main` = `caa64758296a4eb4d388d5c19308328719ed56fb`. 이 워크트리의 `HEAD` 도 같은 커밋이다.
- 원격 둘 다 `dccd5b80f9bff19f14232ff60b8b5b221944a2d8` 에 멈춰 있다 — GitHub `github`(`git@github.com:nacyot/vllm-ds4f-gb10.git`) 의 `dsv41-gb10`, 홈랩 Forgejo `origin`(`git@git.tail39057.ts.net:nacyot/vllm-ds4f-gb10.git`) 의 `main`.
- `git merge-base --is-ancestor dccd5b80f9 main` 통과, 미푸시 49 커밋. **둘 다 fast-forward 다.**
- 4 노드 `~/vllm-dsv41` 은 head `88744c769e` / f323 `d77882fc7` / 37cc `3d81648af` / 27c4 `fe5e0aff1`, 네 대 모두 `git status --short` 0 줄.
- 리모트: head 에는 이미 `fork`(= `https://github.com/nacyot/vllm-ds4f-gb10.git`) 가 있고, 워커 3 대는 `origin`(업스트림 vllm-project) 만 있다.
- 워커에서 GitHub 접근이 된다 — f323 에서 `git ls-remote --heads https://github.com/nacyot/vllm-ds4f-gb10.git dsv41-gb10` 이 현재 sha `dccd5b80f9` 를 돌려줬다. 즉 **레시피의 "워커가 fork 를 fetch" 절차가 그대로 성립**하고 head 를 경유하는 우회가 필요 없다.
- `deploy/gb10-cluster/dsv41` 블롭 대조(main ↔ 노드):
    - f323: 16 개 파일만 다르다 — `bench.py bench2.py decode_gap_probe.py divergence.py kvoff_concurrent.py kvoff_probe.py memlog.py prefill_probe.py smoke.py README.md` 가 옛 판이고 `clock_ctl.sh clock_summary.py kvfs_verify.py test_headroom.py test_kvfs_verify.py test_memlog.py` 가 없다.
    - head: 4 개만 다르다 — `README.md` 옛 판, `clock_ctl.sh clock_summary.py test_headroom.py` 없음.
    - **노드에만 있고 main 에 없는 파일은 네 대 모두 0 개다.** 따라서 `git checkout FETCH_HEAD -- <경로>` 만으로 충분하고 `git rm` 이 필요 없다.
- 유닛이 실제로 읽는 `dsv41_unit.sh dsv41_lib.sh dsv41.env dsv41_ctl.sh serve-node.sh dsv41_warmup.py dsv41_watchdog.sh kvfs_gc.sh` 와 `systemd/` 하위는 **main 과 블롭까지 동일**하다. 이것이 이번 작업의 핵심 불변 조건이다 — sync 가 서비스 동작을 바꿀 수 없다.
- 노드 `.notes/` 에는 `2026-09-12-issue-24-frontend-split/plan.md` 하나만 있고 tracked 다. 이번 sync 에서 `.notes` 는 제외하므로 그대로 남고, main 의 `.gitignore` 화이트리스트(09-16 디렉터리 4 줄)가 노드에 들어가도 노드에 그 디렉터리가 없어 `git status` 에 영향이 없다.

## 1. 현재 동작과 문제점

원격 두 곳은 09-13 README 커밋(`dccd5b80f9`) 에 멈춰 있고, #32~#46 의 성과(영속 유닛, kvfs 검증, Grafana 정정, 인덱서 분할, prefault 비동기, SHORT_RESERVE, 새 기준선, 보고서) 가 하나도 공개돼 있지 않다. 저장소를 보는 사람에게 포크의 현재 상태가 49 커밋만큼 틀리게 보인다.

노드 4 대는 main 커밋을 체리픽한 별도 히스토리라 엔진 트리는 같지만 문서와 보조 스크립트가 뒤진다. 구체적으로:

- `README.md` 가 업스트림 vLLM 원문 그대로다 — 포크 README 로 바뀐 적이 없다. `README.ko.md` 는 아예 없다.
- `deploy/gb10-cluster/dsv41/README.md` 가 374 줄 뒤져 #32 영속 유닛 절, kvfs 검증 절, #33 Grafana 절이 없다. **노드에서 운영 문서를 열어 본 사람이 유닛 운영 규칙을 못 찾는다** — 이게 실질 피해다.
- `.gitignore` 에 `.notes` 화이트리스트 4 줄이 없다.
- `vllm/v1/kv_offload/tiering/manager.py` 가 main `0552479e2e` 의 `@override` 위치 수정 2 줄만큼 다르다. 동작은 동일하다.
- head 에 `clock_ctl.sh clock_summary.py test_headroom.py` 가 없고, 워커 3 대는 여기에 더해 벤치·프로브 스크립트 9 개가 옛 판이고 `kvfs_verify.py test_kvfs_verify.py test_memlog.py` 가 없다.

## 2. 바꿀 것과 접근

### 2-1. main push (워크스테이션)

```bash
git push github main:dsv41-gb10
git push origin main:main
git ls-remote --heads github dsv41-gb10
git ls-remote --heads origin main
```

순서대로, `--force` 없이. 둘 다 fast-forward 라 거절되면 그 자체가 정지 신호다 — 강제하지 않고 보고한다. 푸시 뒤 두 원격이 `caa6475829` 로 같은지 `ls-remote` 로 확인한다.

README 내용은 손대지 않는다. "다른 엔진 비교를 README 에 넣지 않는다" 는 오너 규칙은 이미 지켜져 있고, #42 에서 실측 3 행만 추가됐다.

### 2-2. 노드 sync (4 대, 노드별 한 커밋)

푸시가 끝난 뒤에 한다 — 노드가 GitHub 에서 `caa6475829` 를 받아야 하기 때문이다. 노드마다:

```bash
cd ~/vllm-dsv41
git remote add fork https://github.com/nacyot/vllm-ds4f-gb10.git   # 워커 3 대만, head 는 이미 있음
git fetch fork dsv41-gb10
git diff --stat HEAD FETCH_HEAD -- . ':!.notes'                     # 게이트
git checkout FETCH_HEAD -- README.md README.ko.md .gitignore \
    deploy/gb10-cluster/dsv41 vllm/v1/kv_offload/tiering/manager.py
git commit -m "deploy(dsv41): sync docs, deploy scripts and the @override fix from main caa6475829"
```

**게이트**: `git diff --stat HEAD FETCH_HEAD -- . ':!.notes'` 가 0-1 절에서 확인한 목록(노드별 최대 21 경로) 밖의 파일을 보이면 거기서 멈추고 보고한다. 엔진 트리가 갈라졌다는 뜻이고, 그 경우 이 절차로 덮으면 안 된다.

한 대씩 순서대로(head → f323 → 37cc → 27c4) 하고, 각 대에서 완료 기준을 확인한 뒤 다음으로 넘어간다.

### 2-3. 저장소 노트

`.notes/2026-09-16-issue-35-epic-wrapup/{plan.md, validation.md}` 와 `.gitignore` 에 `!.notes/2026-09-16-issue-35-epic-wrapup/` 한 줄(215~218 행 관례 그대로). 워크트리 커밋은 이 노트뿐이다.

## 3. 영향 사이트 (호출자·소비자)

- **실행 중인 서버**: 영향 없음. 유닛이 읽는 스크립트 8 개와 `systemd/` 가 블롭까지 동일하고(0 절), `manager.py` 는 이미 import 된 모듈이라 파일 교체가 실행 중 프로세스에 반영되지 않는다. 다음 재기동 때 main 과 같은 코드가 올라오며, 그 코드는 `@override` 위치만 다른 동작 동일 판이다.
- **설치된 systemd 유닛**(`~/.config/systemd/user`): 저장소의 `systemd/` 는 템플릿이고 설치본과 별개다. 게다가 블롭이 동일해 템플릿조차 바뀌지 않는다.
- **`dsv41-override.env`**: 건드리지 않는다. 골든룰상 0 bytes 여야 한다.
- **head 의 워치독 타이머**: 이번 작업은 벤치가 없어 stop/start 가 필요 없다. 그대로 둔다.
- **노드에서 스크립트를 직접 돌리는 사람**: 워커 3 대의 벤치·프로브 9 개가 최신판으로 바뀐다. 전부 head 에서만 쓰는 스크립트라 실사용 경로에 없지만, 바뀐다는 사실은 validation 에 적는다.
- **공개 저장소를 보는 사람**: GitHub 기본 브랜치 `dsv41-gb10` 이 49 커밋 앞으로 간다. `gb10-longctx-offload`(V4.0) 브랜치는 건드리지 않는다.

## 4. 검증 방법

노드별 완료 기준:

- `git diff --stat HEAD FETCH_HEAD -- . ':!.notes'` 가 **빈 출력**.
- `git status --short` 가 **0 줄**.
- 커밋 sha 기록.

push 검증:

- 푸시 전후 sha, `git ls-remote --heads` 로 두 원격이 `caa6475829`.

골든룰 재확인(sync 직후 4 노드, #42 validation 과 같은 표 형식):

| 확인 항목 | 기대값 |
| --- | --- |
| `dsv41-serve` | active |
| 유닛 `Environment` | 비어 있음 |
| `dsv41-override.env` | 0 bytes |
| `gpu-clock-cap.service` | active, `-lgc 300,2000` |
| SM 클럭 | 1989 MHz |
| `~/vllm-dsv41` dirty | 0 줄 |

head 추가: `dsv41-watchdog.timer` active, `http://127.0.0.1:8888/health` 200, ATTENTION 없음, MemAvailable 기록.

이 전부를 `.notes/2026-09-16-issue-35-epic-wrapup/validation.md` 에 푸시 전후 sha / 노드별 sync 커밋 sha / 빈 `git diff --stat` 출력 / 골든룰 표로 남긴다.

## 5. 금지 사항

레시피가 명시한 것을 그대로 옮긴다. 이유가 분명한 것만 덧붙였다.

- `git reset --hard`, 히스토리 재작성, `--force` 푸시 — 노드 히스토리는 체리픽본이라 리셋하면 복구 경로가 없다.
- 서비스 stop/restart, `dsv41-override.env` 편집, `nvidia-smi` 클럭 명령(`-pm/-rgc/-lgc`), 파이썬 재설치.
- `.notes/` 를 노드에 동기화하지 않는다.
- 이슈 본문 수정과 이슈 닫기 — 매니저 몫이다.

## 6. 이번에 하지 않을 것

- **머지 뒤 재푸시.** 이 노트 커밋은 push 뒤에 main 으로 들어가므로 원격은 노트 한두 커밋만큼 뒤진 상태로 끝난다. 레시피가 명시적으로 허용한 상태이고, 다음 push 때 올라간다. 머지 단계 뒤 추가 푸시를 이 태스크에서 하지 않는다.
- **오너 결정 대기 항목** — 되묻지 않고 닫을 때 인계한다: 옛 저장소 `~/dsv41-prep/kvfs` 37 GiB(4 노드), f323 도커 이미지 `dsv41-4x-spark:local` 33.2 GB 와 dangling `<none>` `171c04994a8d`, #45 워커 NFS fstab.
- **벤치·프로파일 재실행.** 최종 수치는 비교 보고서 v6 와 `.notes/2026-09-16-issue-46-final-baseline/results.md`, `.notes/2026-09-16-issue-39-gate-ab/results.md` 로 확정돼 있다.
- **메모리 파일 갱신** — 매니저 몫.

## 7. 오너 확인이 필요한 것

없다. 레시피가 범위·절차·금지사항을 모두 정했고, 방향이 갈릴 수 있었던 유일한 지점(워커의 GitHub 접근 가능 여부)은 0 절에서 실측으로 닫았다. 오너 결정 대기 항목 3 건은 6 절대로 되묻지 않고 인계한다.
