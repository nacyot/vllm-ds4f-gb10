# 이슈 #43 검증 기록 (2026-09-16)

계획과 결과 대장은 `plan.md`. 범위 정본은 #43 매니저 코멘트 `icmt-01ae8f64`.
**엔진 코드 변경 0, 서버 재기동 0, 벤치 0, pytest 0, 노드 sync 0, 삭제 명령 0.**

## 1. 코드 변경 0

```text
$ git diff --stat caa6475829 HEAD -- . ':!.notes' ':!.gitignore'
(빈 출력)
```

건드린 파일은 `.gitignore` 한 줄(노트 디렉터리 negation)과 `.notes/2026-09-16-issue-43-meta-close/` 뿐이다. `deploy/`, `vllm/`, `dsv41.env`, 오버라이드 파일, 노드 트리는 열지 않았다.

## 2. 산출물

| 파일 | 내용 |
| --- | --- |
| `plan.md` | 승인 계획 + 결과 대장(주제별 "무엇을 물었나 → 무엇을 했나 → 결과 → 증거 파일") |
| `validation.md` | 이 파일 |
| `report.md` | 마감 보고서 원본(마크다운) |
| `report.html` | 같은 내용을 self-contained HTML 로 렌더, hangar 업로드본과 바이트 동일 |

## 3. 숫자 대조 (근거 노트 = 정본)

결과 대장의 수치는 매니저 코멘트 표가 아니라 각 근거 노트 `results.md` 에서 다시 읽었다. 대조 결과 **두 곳이 코멘트와 어긋난다. 노트 값이 정본이므로 코멘트를 고쳐야 한다.**

### 3-1. 정정이 필요한 항목

| 위치 | 코멘트 표기 | 노트 정본 | 설명 |
| --- | --- | --- | --- |
| #37 행 | "코드 c1 스텝 75.9 → 68.2 ms" | **70.6 → 68.2 ms** | 75.9 ms 는 #37 의 기준선이 아니라 09-14 비교군 `fork0914` 의 값이다. #37 자체 기준선(부팅 0)은 73.3 / 70.2 / 68.4 이고 노트 제목이 "70.6 → 68.2" 로 적고 있다. 근거: `.notes/2026-09-15-issue-37-engram-prefault/results.md` 1 행, 2 절 대조표 |
| #34 행 | "짧은 요청 65 → 7~9 s" | **64.50 / 68.01 → 7.01 / 8.71 s** | 기준선이 두 회 측정이고 65 s 단일값이 아니다. 근거: `.notes/2026-09-15-issue-34-hol-reserve/results.md` 1 절 판정표 |

두 건 모두 방향과 결론은 같고 어긋난 것은 기준선 표기다. 코멘트를 직접 고치지 않았다.

### 3-2. 일치를 확인한 항목

| 항목 | 코멘트 | 노트 정본 | 출처 |
| --- | --- | --- | --- |
| #40 미채택 | +0.3 % | 0.304 % | issue-40 `results.md` 상태 절 |
| #41 귀속 | 26.4 중 MoE 19.8(라우팅 15.4), 호스트 0.2 미만 | 동일 (+19.78, 라우팅 +15.42, 호스트 전 항목 0.2 미만) | issue-41 `results.md` 2-1 표 |
| #38 분할 | S128 +22~23 % | +22.41 % / +23.11 % | issue-39 `results.md` 결론 |
| #17 | NCCL 대기 73 % 는 Engram 조회 랭크 차, mHC 여지 없음 | 동일 (불균형 583 ms 중 425 ms, mHC 하한 68 / 융합 상한 66.4 ms 대 T 66.7) | issue-17 `results.md` 1 절, 2-1 표 |
| #36 | AG4 208 s, 두 노브 기본값 | 208.0 s, 헤드 최저 3.67 GiB | issue-36 `results.md` 1~2 절 |
| #32 | 첫 8K TTFT 21 → 5.3 s | 21.0 → 5.34 s | issue-32 `results.md` 1 절 표 |
| #33 | local_compute 기준, 요청당 속도 72 | 패널 72 가 1,867.3 대 실측 1,845.3 tok/s (+1.2 %) | issue-33 `results.md` 판정표 |
| #10 | 라이브 1,045,649 파일 불일치 0 | files=1045649 ok=1045649, 전 항목 0, deleted=0 | issue-10 `results.md` 191 행 |
| 최종 수치 | 94 / 97 / 82 / 80 / 102 | 94.37 / 96.54 / 81.54 / 80.10 / 102.29 | issue-46 `results.md` 7 행, 135 절 |
| #31 | 오너 결정 (d) 현상 유지 | 이슈 status done | `ryno issue view 31` |

### 3-3. 대장에 새로 넣은 값 (코멘트에 없던 것)

| 값 | 출처 |
| --- | --- |
| `prod-final` 코드 c1 스텝 67.7 ms, fork0914 75.9, SGLang 65.4 | issue-46 `results.md` 스텝 표 76 행 |
| S128 3 회 1,598.9 / 1,633.3 / 1,683.7, 평균 1,638.6 | issue-46 `results.md` S128 편차 표 |
| 긴 세션 158.5K 콜드 1,933.6 tok/s (SGLang 의 98.06 %) | issue-46 `results.md` 긴 세션 표 |
| HOL `prod-final` 14.55 s | issue-46 `results.md` 요약 9 행 |
| 헤드 관측 최저 fork0914 4.64 → prod-final 7.71 GiB | issue-46 `results.md` 헤드 여유 표 |
| #17 게이트 레버 비프로파일 S128 1,733 대 1,737 tok/s | issue-17 `results.md` 1 절 |
| #37 모드 2 산문 페널티 +3.7 ms | issue-37 `results.md` 1 절 |
| #34 첫 부팅 무효과 64.96 / 66.73 s 와 `input_budget - draft_slots` 원인 | issue-34 `results.md` 2 절 |
| #32 복구 시간 241 / 302 s 와 약 4 분 | issue-32 `results.md` 1 절 표 |

## 4. #43 본문 갱신

`ryno issue edit '#43' --body-file` 로 **"현재 상태와 #35의 관계" 절만** 교체했다. 갱신 뒤 `ryno issue show '#43' --json | jq -r .body` 로 받아 확인했다.

| 확인 | 결과 |
| --- | --- |
| 보낸 본문과 저장된 본문 | 끝의 빈 줄 하나를 빼고 동일 |
| "최초 등록 당시 주제와 배경" 이하 원문 | 갱신 전 본문 11 행 이하와 diff 0 (끝 빈 줄 제외) |
| 새 절 구성 | 다섯 주제 결과 표, 절차 4 건, 최종 수치 한 줄, 남은 것 3 항목, 결과 대장과 보고서 경로 |
| 이슈 status | **갱신 전후 모두 `in_progress`. 건드리지 않았다** |

status 에 관한 기록: 매니저 완료 기준은 "status 는 open 그대로" 인데 Ryno 실제 값은 갱신 전에도 `in_progress` 였다. 이 작업에서 상태를 바꾸지 않았으므로 갱신 전 값이 그대로 남아 있다. done 전환은 매니저가 한다.

## 5. 마감 보고서와 hangar

`report.md` 를 `report-writing` 스킬의 기록/done 모드로 쓰고 self-contained HTML 로 렌더했다. 오너 원 보고서 `dsv41-meta-issue-43-2026-09-15` 의 다섯 주제 순서를 따랐고 주제마다 "당시 → 지금" 표를 넣었다. 외부 스크립트와 폰트가 없고 줄표와 가운뎃점을 쓰지 않았다. README 는 손대지 않았으므로 타 엔진 비교 수치는 보고서에만 있다.

```text
$ hangar upload -slug dsv41-meta-issue-43-close-2026-09-16 -type 리포트 report.html
uploaded /p/dsv41-meta-issue-43-close-2026-09-16/ (id=dsv41-meta-issue-43-close-2026-09-16,
         files=1, status=active, version=1)
title adopted from the document: "메타 이슈 #43 마감: 다섯 주제가 어떻게 끝났나"
verified ... serves the same bytes as report.html (sha256 51ddd64ea650, 23.0 KiB)

$ hangar upload -slug dsv41-meta-issue-43-close-2026-09-16 -type 리포트 --replace report.html
replaced /p/dsv41-meta-issue-43-close-2026-09-16/ (status=active, version=2)
verified ... serves the same bytes as report.html (sha256 e7956605cbd0, 23.1 KiB)
```

v2 는 검증 단계에서 찾은 본문 오류 두 건을 고친 판이다(아래 8 절). 같은 세션에서 몇 분 전 올린 같은 슬러그를 고친 것이라 남의 문서를 덮지 않았다.

`hangar info` 확인: status `active`, type `리포트`, format `html`, version `v2`, 첫 업로드 2026-09-15T21:01:38Z(= 09-16 06:01 KST).
URL: <https://hangar.tail39057.ts.net/p/dsv41-meta-issue-43-close-2026-09-16/>

새 슬러그다. 업로드 전 `hangar info` 가 404 였으므로 기존 문서를 덮지 않았다.

## 6. head 골든룰 (읽기만, 2026-09-16 06:02:09 KST)

| 항목 | 값 |
| --- | --- |
| `dsv41-serve` | active |
| 유닛 `Environment` | 비어 있음 |
| `~/dsv41-prep/dsv41-override.env` | 0 bytes |
| `gpu-clock-cap.service` | active |
| SM 클럭 | 1989 MHz (최대 3003) |
| `http://127.0.0.1:8888/health` | 200 |
| `dsv41-watchdog.timer` | active |
| MemAvailable | 9.91 GiB |

`nvidia-smi` 는 조회 옵션만 썼다. `-pm`, `-lgc`, `-rgc` 와 `gpu-clock-cap.service` 조작은 없다. 워치독 타이머는 멈추지 않았다.

## 7. push (머지 뒤 기록)

머지 단계 뒤에 실행하고 이 절에 전후 sha 를 채운다.

| 대상 | push 전 sha |
| --- | --- |
| 로컬 `main` | `dfc7f53104` |
| GitHub `github` `dsv41-gb10` | `caa6475829` |
| Forgejo `origin` `main` | `caa6475829` |

`git merge-base --is-ancestor caa6475829 main` 통과, 미푸시 1 커밋으로 둘 다 fast-forward 다. `--force` 는 쓰지 않는다. 노드 sync 는 하지 않는다.

## 8. 검증 단계에서 고친 것

`git diff dfc7f53104..HEAD` 를 계획과 요청 원문에 대조하면서 `report.md` 의 수치를 근거 노트와 다시 맞췄다. 실질 오류 두 건을 그 자리에서 고쳤다.

| 위치 | 고치기 전 | 고친 뒤 | 근거 |
| --- | --- | --- | --- |
| 리드 문단 | "열두 건이 전부 닫혔고", "산문 94.4, 코드 94 대 97" | "열두 건 중 열한 건이 닫혔다", "산문 c1 94.4, 코드 c1 96.5, 코드 c4 합산 81.5" | #29 가 열려 있으므로 12 건 중 11 건이다. 점수 다섯 개는 issue-46 `results.md` 135 절의 94.37 / 96.54 / 81.54 / 80.10 / 102.29 이고 "94 대 97" 은 어느 값도 아니었다 |
| 주제 2, 동일성 검증 문단 | "8K 프롬프트는 첫 토큰이 거의 동점이라(logprob 차 0.3 nats 안)" | 8K 는 두 후보 logprob 이 회차마다 뒤바뀔 만큼 가깝다고 쓰고, 32K 는 차순위와 4.5 nats 이상 벌어진다고 구분 | 0.13~0.35 nats 는 8K 첫 토큰이 아니라 32K 공통 접두 top-1 의 차다. 8K 첫 토큰 차는 0.5~1.0 nats 다. 근거: issue-38 `results.md` 4 절 greedy 대조군 |

고친 뒤 `report.html` 을 다시 렌더하고 hangar 를 v2 로 교체했다. `plan.md` 의 결과 대장에는 같은 오류가 없었다(8K 는 "near-tie 라 판정에 못 쓴다" 로만 적혀 있었다).

인용 경로 점검도 했다. `plan.md`, `report.md`, `validation.md` 가 가리키는 `.notes/**` 경로 14 개가 모두 실재한다.
