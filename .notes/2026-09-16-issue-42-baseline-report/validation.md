# 이슈 #42 검증 기록 (2026-09-16)

계획: `plan.md`. 범위 정본: #42 매니저 코멘트 `icmt-ef080914`.
서버는 켜진 채로 두었고 재기동·벤치·엔진 코드 변경은 하지 않았다.

## 1. 보고서 v5

- 원본 보존: `curl` 로 v4 를 받아 `report-v4.html`(67,681 바이트, 1,467 줄, 표 21 개)로 저장.
- 편집본: `report-v5.html`. 표와 절 구성은 v4 그대로이고 표 2 하나가 빠져 표 20 개다.
- 업로드와 확인:

```text
$ hangar upload -replace -slug dsv41-sglang-vs-vllm-2026-09-14 -type 리포트 report-v5.html
replaced /p/dsv41-sglang-vs-vllm-2026-09-14/ (id=dsv41-sglang-vs-vllm-2026-09-14, status=active, version=5)
verified /p/dsv41-sglang-vs-vllm-2026-09-14/ — serves the same bytes as report-v5.html (sha256 413119b92856, 72.7 KiB)

$ hangar info dsv41-sglang-vs-vllm-2026-09-14      # 전문은 hangar-info-v5.txt
  version         v5
  status          active
  type            리포트
  url             https://hangar.tail39057.ts.net/p/dsv41-sglang-vs-vllm-2026-09-14/
```

### 정정 항목별 확인

| # | 정정 항목 | v5 에서 확인한 것 |
| --- | --- | --- |
| 1 | 표 1 기대치 열 → `prod-final` 실측 | 열 이름 "우리 포크 prod-final", 값 86.0 / 194.2 / 45.2 / 1,881 / 1,684(범위 1,599~1,894) / 191.9 / 0.80 / 14.6 / 7.71. "우리 포크 현재" 열은 "우리 포크 09-14" 로 개명 |
| 2 | 표 2 "기대치가 어디서 오나" 삭제 | 캡션 `표 2.` 가 문서에 없다(표 21 → 20). 나머지 표 번호는 v4 그대로 두어 2 번 자리가 비어 있고, 그 사실을 정정 절에 적었다 |
| 3 | "프리필의 나머지는 메모리 예산 문제" 철회 | 해당 문장 삭제, 같은 서빙 인자에서 T2W 1,522 대 2,034 로 갈린 근거와 인덱서 분할 +22.4% / +23.1% 로 대체 |
| 4 | "개선 계획" 표 | "T2W Engram 준비 단계 이식" 행 삭제(본문에는 정정 절의 인용 한 번만 남는다). 표를 결말·실측 몫·남은 위험 열로 다시 써서 채택 5, 미채택 3, 보류 1, 미착수 1 을 적었다 |
| 5 | 긴 프리필 중 대기 88.9 s | 88.9 는 09-14 당시 값으로 문장을 과거형으로 바꾸고, 해결 문단을 붙여 7.01 / 8.71 s 와 `prod-final` 14.6 s(그 회차 프리필 1,418 tok/s), SGLang 56.4 s 는 프리필 2,102 tok/s 의 결과임을 적었다. 표 1 값도 14.6 |
| 6 | 동시 2 격차 = 호스트 지문 | 철회. MoE 그룹 GEMM 19.8 ms(라우팅 15.4), 호스트 0.2 ms 미만, 같은 프롬프트 동시 2 벤치 폐기와 이봉 설명 추가 |
| 7 | NCCL 대기 = 느린 랭크 클럭 | v4 본문에 그 문장 자체는 없었다. 정정 절에 해석 철회를 적고, 개선 계획 9 번(RoCE one-shot all-reduce) 행에 128K 청크 NCCL 대기의 73% 가 Engram 조회 랭크 차(r0 46 → r3 468 ms)이고 mHC 가 대역폭 하한의 1.24~1.37 배라 여지가 없다는 것을 넣어 우선순위를 내렸다 |
| 8 | 코드 c1 스텝 75.9 ms | 진단 문단을 과거형으로 바꾸고 68.2 ms, `prod-final` 67.7 ms, 처리량 86.0 tok/s 를 적었다. A/B 표 안의 75.9 는 그때 실제로 잰 값이라 그대로 둔다 |
| 9 | Grafana 단서 | v4 에 Grafana 인용 문장이 0 곳이라 본문에 새로 만들지 않고, 정정 절 끝에 "09-16 01:35 이전 값은 복원 포함(옛 식)" 한 줄로 남겼다 |
| 10 | 헤드 여유 레버(#31) | 오너 결정 (d) 현상 유지로 적고, 그래서 b12x 계열이 채택 후보에서 빠졌다고 표와 본문 양쪽에 썼다 |
| 11 | S128 범위, 비교 상대 | 정정 절에 프로덕션 구성 범위 1,599~1,894 tok/s(회차 편차 ±8%), 동시 4/8 비교 상대는 `combo-nob12x`, 구성 A~C 는 상대 몫에만 쓴다고 명시 |

추가로 머리말의 "측정에 쓴 노브는 하나도 반영하지 않았다" 와 "03:41 프로덕션 복구 완료" 는 v5 시점 사실(채택 노브가 `dsv41.env` 기본값, 프로덕션이 override 0 B 로 그 기본값 사용)로 고쳤다. 새로 쓴 문장에 줄표와 가운뎃점은 쓰지 않았고(문서 전체 0 개), HTML 태그 균형과 표 수를 스크립트로 확인했다.

## 2. README 연대기

`README.md` 와 `README.ko.md` 를 같은 내용으로 고쳤다.

- 실측 표에 09-16 행 3 개 추가: 128K 프리필 회차 범위 1,599~1,894 tok/s, 디코드 1 스트림 산문 45.2 / 코드 86.0 tok/s, 128K 프리필 뒤에 밀린 짧은 요청 7~15 초. 타 엔진 수치는 넣지 않았다.
- "Added in this fork / 이 포크에서 추가한 것" 표: 스케줄러 행에 짧은 요청 예약, Engram 행에 디코드 스텝 비동기 프리폴트, GB10(SM121) 행에 인덱서 프리필 TP 분할.
- 계획에 없던 한 곳: Engram 설명 글머리의 "3스텝 뒤 페이지를 놓습니다 / pages are released after 3 steps" 는 `ENGRAM_RELEASE=0`(기본값, #36) 과 어긋나 "회수는 커널에 맡깁니다 / reclaim is left to the kernel" 로 고쳤다.
- 두 파일 대조: "측정값" 절이 각각 13 줄이고 숫자 열이 일치한다(한국어 쪽의 여분 2 와 7 은 "세션 2개", "세션 7개" 표기 차이).

## 3. 노드 잔존물 삭제

매니저 목록만, 노드 한 대씩 전경에서, 절대 경로를 그대로 적어 지웠다. 변수·글롭 조합과 `find -delete` 는 쓰지 않았다. 각 대상은 삭제 전 `ls -ld` 로 확인했고 전부 존재했으며, 삭제 뒤 `[ -e ]` 로 부재를 확인했다.

데이터 파일은 지우기 전에 저장소 사본을 확인했다.

| 파일 | 판정 |
| --- | --- |
| `~/sglang-cmp/results/i39.log` | `.notes/2026-09-16-issue-39-gate-ab/i39.log` 와 바이트 동일 |
| `~/dsv41-prep/decodebench_i41.py`, `profile_i41.sh` | `.notes/2026-09-15-issue-41-c2-decode-gap/*.txt` 와 바이트 동일 |
| `~/dsv41-prep/decode_i41.jsonl` | 45 행 전부가 #41 노트의 `implement-`(26) 와 `plan-probe-`(19) 합집합. 누락 0 |
| `~/dsv41-prep/bench/decode-i32.jsonl` | 저장소 사본 없음 → `head-rescue/decode-i32.jsonl` 로 복사 |
| `~/dsv41-prep/i34/logs-base.json`, `mixlong_i34.sh.txt` | 저장소 사본 없음 → `head-rescue/` 로 복사 |

### head gx10-6040

`git -C ~/vllm-dsv41 worktree list` 에 `t2w-best`, `t2w-boot10` 이 e47aa780bc detached 로 등록돼 있었다. `worktree remove --force` 두 번 뒤 `worktree prune`, 빈 `~/t2w-trees` 를 `rmdir`. 남은 워크트리는 `/home/nacyot/vllm-dspark-fork`(ab7723d3ff, V4.0) 와 `/home/nacyot/vllm-dsv41` 둘이고 `vllm-dspark-fork` 는 손대지 않았다.

지운 것(전부 `gone` 확인): `~/t2w-trees`, `~/t2w-cache`, `~/dsv41-prep/prof-i17`, `~/dsv41-prep/prof-i41-base`, `~/dsv41-prep/i34`, `~/dsv41-prep/i10-pytest.0d3aKA`, `~/dsv41-prep/i10-pytest.PDst6K`, `~/vllm-dsv41/.pytest_cache`, `~/vllm-dsv41/deploy/gb10-cluster/dsv41/__pycache__`, `~/dsv41-prep/i40-8fd917c8ae`, `~/dsv41-prep/i38`, `~/dsv41-prep/i38-a`, `~/dsv41-prep/i38-b`, `~/dsv41-prep/trace_ops_i40.py`, `~/dsv41-prep/trace_ri_children_i40.py`, `~/dsv41-prep/trace_steps_i40.py`, `~/dsv41-prep/patches/i17-223024`, `~/dsv41-prep/patches/i17-revert-225245`, `~/dsv41-prep/decode_i41.jsonl`, `~/dsv41-prep/decodebench_i41.py`, `~/dsv41-prep/profile_i41.sh`, `~/dsv41-prep/bench/decode-i32.jsonl`, `~/sglang-cmp/results/i39.log` (23 개).

df `/`: 72G 여유 92% → 73G 여유 92%.

### 워커 gx10-f323

`t2w-best`, `t2w-boot10` 등록돼 있어 `worktree remove --force` 로 지웠다. 그 밖에 `~/t2w-cache`, `~/dsv41-prep/prof-i17`, `~/dsv41-prep/prof-i41-base`, `~/dsv41-prep/i38-a`, `~/dsv41-prep/i38-b`, `~/dsv41-prep/patches/i17-223024`, `~/dsv41-prep/patches/i17-revert-225245`, `~/vllm-dsv41/deploy/gb10-cluster/dsv41/__pycache__` 를 지웠다. 전부 `gone` 확인.

SGLang 이미지: `docker ps -a` 가 비어 있어(컨테이너 0 개) `docker rmi lmsysorg/sglang:dev-dsv41` 실행, `Untagged` 와 `Deleted: sha256:381b27ffa19b` 확인. 목록에서 사라졌다.

df `/`: 89G 여유 90% → 90G 여유 90%.

**예상과 다른 점.** 매니저 목록은 f323 에서 약 35 GB 회수를 예상했는데 실제로는 약 1 GB 다. `docker system df` 가 이미지 3 개 합계 33.2 GB 로 나오고 `docker images` 도 `dsv41-4x-spark:local` 을 같은 33.2 GB 로 표시한다. 즉 SGLang 이미지가 프로덕션 이미지와 레이어를 공유하고 있어서 이미지별 33.2 GB 는 중복 집계였고, 태그를 지워도 config 만 사라진다. 삭제 자체는 안전했고(`dsv41-4x-spark:local` 은 그대로 동작) 실제 여유는 늘지 않았다. 같은 레이어를 쓰는 dangling 이미지 `171c04994a8d` 가 하나 남아 있으나 매니저 목록에 없어 손대지 않았다.

### 워커 gx10-37cc

등록된 워크트리 2 개 `worktree remove --force`, 공통 목록 7 개 삭제, 전부 `gone` 확인. df `/`: 138G 여유 85% → 139G 여유 85%.

### 워커 gx10-27c4

같은 절차. 전부 `gone` 확인. df `/`: 122G 여유 87% → 124G 여유 86%.

### 지우지 않은 것 (삭제 뒤 존재 확인)

- 4 노드: `~/dsv41-prep/kvfs`(오너 결정 대기), `~/vllm-dsv41` 트리와 venv.
- head: `~/vmagent/scrape.yml`, `~/migration-027`, `~/ds4f-logs`, `/home/nacyot/vllm-dspark-fork`, `~/sglang-cmp/results/{casebench,decode,longctx}.jsonl`, `~/dsv41-prep/bench`(안의 `decode-i32.jsonl` 한 개만 지웠다), 워치독 state, 구 유닛 파일.
- f323: `dsv41-4x-spark:local` 이미지.
- `kvfs_verify.py` 는 돌리지 않았다. 지운 것이 KV 저장소와 무관하다.

## 4. 노드 골든룰 (삭제 뒤)

| 노드 | dsv41-serve | Environment | override.env | gpu-clock-cap | SM 클럭 | `vllm-dsv41` dirty |
| --- | --- | --- | --- | --- | --- | --- |
| gx10-6040 | active | 비어 있음 | 0 bytes | active | 1989 MHz | 0 줄 |
| gx10-f323 | active | 비어 있음 | 0 bytes | active | 1989 MHz | 0 줄 |
| gx10-37cc | active | 비어 있음 | 0 bytes | active | 1989 MHz | 0 줄 |
| gx10-27c4 | active | 비어 있음 | 0 bytes | active | 1989 MHz | 0 줄 |

`gpu-clock-cap.service` 는 네 대 모두 `nvidia-smi -lgc 300,2000` 으로 그대로다.

head 추가: `dsv41-watchdog.timer` active, `http://127.0.0.1:8888/health` 200, MemAvailable 9.57 GiB.

`.pytest_cache` 와 `__pycache__` 는 `.gitignore` 대상이라 삭제 전에도 `git status --porcelain` 에 뜨지 않았고, 삭제 뒤에도 네 노드 모두 0 줄이다.

## 5. 저장소 변경

코드 변경 0. 건드린 파일은 `.gitignore`(노트 디렉터리 negation 한 줄), `README.md`, `README.ko.md`, 그리고 `.notes/2026-09-16-issue-42-baseline-report/` 다.

`pre-commit` 은 커밋마다 돌았다. `markdownlint-cli2` 가 첫 커밋 때 `plan.md` 의 목록 앞 빈 줄을 고쳐 다시 스테이징했고, 두 번째 시도에서 전 훅 통과했다.

## 6. 이번에 하지 않은 것

노드 README 헝크 sync 와 `main` push(#35 마무리), 옛 `~/dsv41-prep/kvfs` 삭제(오너), 메모리 파일 갱신(매니저), 서버 재기동·벤치·엔진 코드 변경.
