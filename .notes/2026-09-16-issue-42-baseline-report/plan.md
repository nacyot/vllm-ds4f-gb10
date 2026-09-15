# 이슈 #42 계획: 비교 보고서 v5 정정, README 연대기, 노드 잔존물 삭제

작성 2026-09-16, 워크트리 `issue-42-baseline-report`, 기준 main `aa4e19ceee`.
범위 정본은 #42 매니저 코멘트 `icmt-ef080914` 와 인계 코멘트 9 개다. 이 계획은 그 범위를 실행 순서로 옮긴 것이고, 범위를 넓히지 않는다.

## 0. 현재 상태 (계획 단계에서 실측한 것)

방향이 갈릴 수 있는 지점만 확인했다. 결과는 전부 매니저 코멘트와 일치한다.

- hangar CLI `/Users/nacyot/.local/bin/hangar` 동작, `hangar info dsv41-sglang-vs-vllm-2026-09-14` 가 **version v4**, format html, status active 를 돌려준다.
- v4 공개 URL `curl` 200, 67,681 바이트 자립형 HTML(1,467 줄, 표 21 개). h2 아홉 절: 네 축 한눈 비교 / 무엇을 쟀나 / 결과 / 격차는 어디에서 오나 / 우리 포크에서 돌린 A/B / 개선 계획 / 공개 수치를 어떻게 읽을 것인가 / 한계와 실패한 실험 / 부록.
- 정정 대상 문자열이 v4 에 실제로 있다: "우리 포크 기대치" 열과 "표 2. 기대치가 어디서 오나", "프리필의 나머지는 엔진 문제가 아니라 메모리 예산 문제다", "88.9"(2 곳), "75.9"(7 곳), NCCL(6 곳), "T2W Engram 준비 단계". "Grafana" 는 v4 에 **0 곳**이라 #33 단서 문장은 붙일 자리가 없다(아래 2-8).
- 원본 md 는 저장소·워크스테이션 어디에도 없다. v5 는 v4 HTML 을 고쳐서 만든다.
- 네 노드 SSH 전부 응답. root 사용률 head 92%(72G 여유), f323 90%(89G), 37cc 85%, 27c4 87%.
- 저장소 `.notes/2026-09-16-issue-46-final-baseline/report-tables.txt` 에 5 비교군 표가 이미 생성돼 있다. 보고서 표는 이 파일을 그대로 인용하면 되고 벤치를 다시 돌리지 않는다.
- `.notes/2026-09-15-issue-41-c2-decode-gap/` 에 `implement-decode_i41.jsonl`, `decodebench_i41.py.txt`, `profile_i41.sh.txt` 사본이 이미 있다. 즉 head 의 `decode_i41.jsonl` / `decodebench_i41.py` / `profile_i41.sh` 는 사본 확인만 하면 바로 지울 수 있다.
- `.gitignore` 는 `.notes/` 를 무시하고 이슈별 디렉터리를 negation 으로 하나씩 푼다(215~217 행). 이번 노트 디렉터리도 같은 방식이 필요하다.

## 1. 문제점

보고서 `dsv41-sglang-vs-vllm-2026-09-14`(hangar v4, 2026-09-14) 는 그 시점의 추정과 잠정 진단 위에 서 있다. 그 뒤 #34, #37, #38, #17, #41, #40, #33, #32, #46, #39 가 끝나면서 여섯 개 문장이 실측과 어긋났다.

1. 표 1 의 "우리 포크 기대치" 열은 노브 하나짜리 변화율을 옮긴 추정인데, 지금은 노브를 다 켠 `prod-final` 실측이 있다.
2. "프리필의 나머지는 엔진 문제가 아니라 메모리 예산 문제다" 는 틀렸다. T2W 기본과 스피드런은 서빙 인자(300K, 동시 8, 배치 8,192, 블록 128)가 같은데 1,522 와 2,034 로 갈린다.
3. "개선 계획" 표의 "T2W Engram 준비 단계 이식" 행은 근거가 사라졌고, 기대치 열 자체가 #39 몫 표로 대체됐다.
4. "긴 프리필 중 대기 88.9 s" 는 `SHORT_RESERVE=4096` 채택(#34) 뒤 7~15 s 다.
5. "동시 2 격차 = 호스트 지문" 은 #41 에서 MoE 그룹 GEMM 으로 귀속됐다.
6. "NCCL 대기 = 느린 랭크 클럭" 은 네 대 1989 MHz 뒤로 맞지 않는다(#17).
7. 코드 디코드 c1 스텝 75.9 ms 는 #37 뒤 68.2 ms, `prod-final` 기준 67.7 ms 다.

같은 이유로 `README.md` / `README.ko.md` 의 실측 표는 09-13 에서 멈춰 있고, "Added in this fork" 표에 이번 에픽에서 채택한 세 가지(`SHORT_RESERVE` 예약, 디코드 비동기 prefault, 인덱서 TP 분할)가 빠져 있다.

노드 쪽은 비교 실험의 잔존물이 남아 있다. head root 92%, f323 90% 이고, f323 의 SGLang 이미지만 33.2 GB 다.

## 2. 보고서 v5

`hangar upload -replace` 로 같은 슬러그에 올린다. 슬러그·제목·구조를 유지하고 내용만 고친다.

절차:

1. `curl -s https://hangar.tail39057.ts.net/p/dsv41-sglang-vs-vllm-2026-09-14/ -o .notes/2026-09-16-issue-42-baseline-report/report-v4.html` 로 v4 원본 보존(계획 단계에서 받아 본 것과 같은 바이트인지 크기로 확인).
2. `report-v4.html` 을 복사해 `report-v5.html` 로 고친다. 표·절 구조와 스타일은 건드리지 않는다.
3. `hangar upload -replace -slug dsv41-sglang-vs-vllm-2026-09-14 -type 리포트 report-v5.html`, `hangar info dsv41-sglang-vs-vllm-2026-09-14` 로 version v5 확인.

문장 규칙: 줄표(—)와 가운뎃점(·)을 쓰지 않는다. 기존 보고서의 문체(평서형 한국어, 표 + 짧은 해설)를 그대로 따른다.

맨 위에 "v5 정정 (2026-09-16)" 절을 새로 두고, 아래 각 항목을 "무엇을 고쳤나 / 왜" 한 줄씩으로 적는다.

### 2-1. 표 1 의 기대치 열 → `prod-final` 실측 열

"우리 포크 기대치(추정)" 열을 `prod-final` 실측으로 교체하고 열 이름도 바꾼다. 수치 출처는 `.notes/2026-09-16-issue-46-final-baseline/report-tables.txt`(5 비교군 표, `sglang-tp4:SGLang t2w-best:T2W vllm-prod:fork0914 full-combo:combo0915 prod-final:prod-final`). 표 1 에 들어가는 대표값: 코드 c1 86.0 tok/s, 산문 c1 45.2, 128K 콜드 프리필 1,683.7, 긴 세션 콜드 1,933.6 / TTFT 81.98 s, 후속 턴 TTFT 0.80 s, 헤드 여유 최저 7.71 GiB.
"헤드 여유 최저" 행의 "3.0 아래로 예상" 은 실측 7.71 로 바뀌고, 본문에 헤드 여유 레버(#31)는 오너 결정 (d) 현상 유지라고 적는다.

### 2-2. 표 2 "기대치가 어디서 오나" 삭제

표와 그 앞뒤 설명("기대치 열은 추정이다 ...", "세 노브는 서로 다른 축을 건드리니 ...")을 "기대치 열은 `prod-final` 실측으로 대체했다" 한 줄로 줄인다.

### 2-3. 2 절 메모리 예산 문단 철회

"프리필의 나머지는 엔진 문제가 아니라 메모리 예산 문제다" 와 뒤따르는 300K/동시 8/8,192/블록 128 설명을 철회 문장으로 바꾼다. 근거: 같은 서빙 인자에서 T2W 기본 1,522 와 스피드런 2,034 가 갈린다. 실제로 확인된 몫은 인덱서 TP 분할로 S128 +22.4% / +23.1%(#39).

### 2-4. "개선 계획" 표 교체

"T2W Engram 준비 단계 이식" 행을 지운다. 기대치 열 대신 #39 몫 표(`.notes/2026-09-16-issue-39-gate-ab/results.md`)를 쓴다.

- 인덱서 TP 분할: S128 +22.4% / +23.1%, S32 이득 없음 (채택, #38)
- 디코드 비동기 prefault: 코드 c1 스텝 −3.5 / −1.1 ms, 산문 c1 +1.0% / +6.2% (채택, #37)
- PR 56562 token_to_req 커널: 게이트 없음, +0.3%, 미채택 (#40)
인용 주의: 구성 A 는 `SHORT_RESERVE=4096`, `SPEC_BLOCK_DROP=0`, `ENGRAM_RELEASE=0` 을 유지한 상태라 09-15 full-combo 로의 완전한 되돌림이 아니다. A 의 S128 1,407 은 kvfs-gc 창이 겹쳤으므로 절대값이 아니라 같은 부팅 안의 상대 몫으로만 인용한다.

### 2-5. 긴 프리필 중 대기

88.9 s(SGLang 56.4) → `SHORT_RESERVE=4096` 뒤 7.01 / 8.71 s(#34), `prod-final` 회차는 14.6 s(그 회차 128K 프리필 1,418.2 tok/s). SGLang 의 56.4 s 는 스케줄러 정책이 아니라 프리필 속도 2,102 tok/s 의 결과라고 함께 적는다. 이 정정은 "긴 프리필 중의 다른 요청: 스케줄러 정책 차이" 절의 진단 문장에도 반영한다.

### 2-6. 동시 2 격차

"호스트 지문(Engram 동기화, 검증 뒤처리 eager)" → MoE 그룹 GEMM 이 +26.4 ms 중 19.8 ms(라우팅 15.4), 호스트 기여 0.2 ms 미만(#41 results.md 2-1, 2-2). 같은 프롬프트 동시 2 벤치는 도착 타이밍에 따라 75 / 93 ms 이봉이라 폐기했고, 동시 2 이상은 서로 다른 프롬프트로 정의한다는 규칙을 한 줄 덧붙인다.

### 2-7. NCCL 대기

"느린 랭크 클럭" → 네 대 1989 MHz 뒤로 맞지 않는다. 128K 청크(2,224 ms) NCCL 대기 원천의 73% 는 Engram 조회의 랭크 차(r0 46 ms → r3 468 ms)다. mHC 호출당 1.72 / 0.84 / 1.06 ms(4,092 토큰, 대역폭 하한의 1.24~1.37 배)이고 09-12 의 796 µs 는 16 토큰 스텝이 섞인 평균이다(#17).

### 2-8. 남은 정정

- 코드 디코드 c1 스텝 75.9 ms → 68.2 ms(#37 `i37`), `prod-final` 은 67.7 ms 이고 처리량은 코드 c1 86.0 tok/s.
- S128 은 단정값 대신 프로덕션 구성 범위 **1,599~1,894 tok/s**(회차 편차 ±8%, #17/#46/#39)로 적는다. `prod-final` 3 회는 1,683.7 / 1,598.9 / 1,633.3, 평균 1,638.6.
- 동시 4/8 디코드의 비교 상대는 `combo-nob12x`(b12x 보류 뒤 기본값)이지 full-combo 가 아니라고 명시한다.
- Grafana 는 v4 본문에 인용이 없다(0 곳). #33 단서는 새로 문장을 만들어 넣지 않고, v5 정정 절에 "Grafana 프리필 패널은 09-16 01:35 에 계산 토큰 기준으로 바뀌었고, 그 이전 값은 복원 포함(옛 식)이다" 를 각주 한 줄로만 남긴다. 이유는 아래 판단 근거.
- 머리말의 "측정에 쓴 노브는 하나도 반영하지 않았다", "03:41 프로덕션 복구 완료" 같은 09-14 시점 상태 문장은 v5 시점 사실(채택 노브가 `dsv41.env` 기본값이고 프로덕션이 override 0 B 로 그 기본값을 쓴다)로 고친다. 고치지 않으면 새 표와 정면으로 어긋나기 때문이다.

판단 근거(에이전트가 정함, 오너 질문 아님): Grafana 인용이 없는데 "09-16 01:35 이전 값은 복원 포함" 만 본문에 새로 넣으면 없는 문장을 고친 꼴이 된다. 매니저 지시가 "인용한 문장이 **있으면**" 이므로 조건이 성립하지 않는다고 보고 정정 절 각주로만 남긴다.

## 3. README 연대기 (두 파일 같은 내용)

`README.md` "Measured" 절(46 행~)과 `README.ko.md` "측정값" 절(46 행~).

- 실측 표에 09-15~16 행 추가: 128K 프리필 1,599~1,894 tok/s(프로덕션 구성 범위), 디코드 산문 c1 45.2 / 코드 c1 86.0 tok/s, 긴 프리필과 겹치는 짧은 요청 7~15 s. 근거 회차는 `prod-final`.
- "Added in this fork" / "이 포크에서 추가한 것" 표: Scheduler 행에 `SHORT_RESERVE` 예약, Engram 행에 디코드 비동기 prefault, 인덱서(GB10/SM121) 행에 TP 분할을 한 줄씩.
- 타 엔진 비교 수치는 README 에 쓰지 않는다(SGLang/T2W 열 없음).
- 영어와 한국어 표의 행 수·수치가 같아야 한다. 마지막에 두 파일의 숫자를 뽑아 대조한다.

## 4. 노드 잔존물 명시 삭제

원칙: 매니저가 05:0x 에 실측한 목록 **그대로만** 지운다. 노드 한 대씩, 전경에서, 절대 경로를 그대로 적어 지운다. 변수·글롭 조합 삭제 금지, `find -delete` 금지. 각 대상은 `ls -ld <절대경로>` 로 먼저 확인하고, 없는 것은 "이미 없음" 으로 기록하고 넘어간다. 서버는 켜진 채 두고 재기동하지 않는다.

순서(노드별로 확인 → 삭제 → 부재 확인):

1. **head gx10-6040**
   - `git -C ~/vllm-dsv41 worktree list` 로 등록 확인 후 `git -C /home/nacyot/vllm-dsv41 worktree remove --force /home/nacyot/t2w-trees/t2w-best`, 같은 방식으로 `t2w-boot10`, 그다음 `worktree prune`, 빈 `/home/nacyot/t2w-trees` 제거. `/home/nacyot/vllm-dspark-fork` 워크트리는 V4.0 트리라 **손대지 않는다**.
   - 디렉터리/파일: `~/t2w-cache/`, `~/dsv41-prep/prof-i17/`, `~/dsv41-prep/prof-i41-base/`, `~/dsv41-prep/i34/`, `~/dsv41-prep/i10-pytest.0d3aKA`, `~/dsv41-prep/i10-pytest.PDst6K`, `~/vllm-dsv41/.pytest_cache`, `~/vllm-dsv41/deploy/gb10-cluster/dsv41/__pycache__`, `~/dsv41-prep/i40-8fd917c8ae/`, `~/dsv41-prep/i38`, `~/dsv41-prep/i38-a`, `~/dsv41-prep/i38-b`, `~/dsv41-prep/trace_ops_i40.py`, `~/dsv41-prep/trace_ri_children_i40.py`, `~/dsv41-prep/trace_steps_i40.py`, `~/dsv41-prep/patches/i17-223024`, `~/dsv41-prep/patches/i17-revert-225245`, `~/dsv41-prep/decode_i41.jsonl`, `~/dsv41-prep/decodebench_i41.py`, `~/dsv41-prep/profile_i41.sh`, `~/dsv41-prep/bench/decode-i32.jsonl`, `~/sglang-cmp/results/i39.log`.
2. **워커 gx10-f323 / gx10-37cc / gx10-27c4** (공통): `~/t2w-trees/{t2w-best,t2w-boot10}`(각 1.6 GB, `git -C ~/vllm-dsv41 worktree list` 로 등록 여부를 먼저 보고 등록돼 있으면 `worktree remove`, 아니면 디렉터리 삭제), `~/t2w-cache/`, `~/dsv41-prep/prof-i17/`, `~/dsv41-prep/prof-i41-base/`, `~/dsv41-prep/i38-a`, `~/dsv41-prep/i38-b`, `~/dsv41-prep/patches/i17-223024`, `~/dsv41-prep/patches/i17-revert-225245`.
3. **f323 만 추가**: `~/vllm-dsv41/deploy/gb10-cluster/dsv41/__pycache__`, SGLang 이미지 `lmsysorg/sglang:dev-dsv41`(33.2 GB). `docker ps -a` 로 그 이미지를 쓰는 컨테이너가 없음을 확인한 뒤 `docker rmi lmsysorg/sglang:dev-dsv41`.

데이터 파일 보호: `.jsonl` 과 `.log`(`decode_i41.jsonl`, `bench/decode-i32.jsonl`, `sglang-cmp/results/i39.log`, `i34/logs-base.json` 류)는 지우기 전에 저장소 `.notes` 에 사본이 있는지 확인하고, 없으면 `.notes/2026-09-16-issue-42-baseline-report/` 로 먼저 복사한다. (계획 단계 확인: `decode_i41.jsonl`·`decodebench_i41.py`·`profile_i41.sh` 는 `.notes/2026-09-15-issue-41-c2-decode-gap/` 에 사본 있음. `i39.log` 는 `.notes/2026-09-16-issue-39-gate-ab/i39.log` 에 있음.)

**지우지 않는 것**(명시): 옛 저장소 `~/dsv41-prep/kvfs`(37 GiB, 4 노드, 오너 결정 대기), `~/vmagent/scrape.yml`, `~/migration-027`, `~/ds4f-logs`, 구 유닛 파일, 워치독 state 디렉터리, `~/sglang-cmp/results/{casebench,decode,longctx}.jsonl`, 노드 `~/vllm-dsv41` 트리와 venv, `/home/nacyot/vllm-dspark-fork` 워크트리.

예상 회수: head 약 1.7 GB, f323 약 35 GB. 37cc / 27c4 는 t2w-trees 3.2 GB + 잡파일.

## 5. 검증

`.notes/2026-09-16-issue-42-baseline-report/validation.md` 에 아래를 그대로 붙인다.

- 보고서: `hangar info dsv41-sglang-vs-vllm-2026-09-14` 의 version v5 출력. 정정 항목 8 개가 v5 본문에 반영됐음을 항목별 grep 결과로 확인(옛 문자열 88.9 / 75.9 / "메모리 예산 문제" / "T2W Engram 준비 단계" / "기대치" 가 정정 뒤에도 남아 있지 않거나, 남아 있다면 정정 절의 인용임을 밝힌다). `report-v4.html` 과 `report-v5.html` 이 노트 디렉터리에 있음.
- README: `README.md` / `README.ko.md` 의 새 행 diff, 두 파일 수치 대조 결과.
- 노드: 노드별 삭제 전 `ls -ld` 목록과 삭제 뒤 `ls` 부재 확인. "지우지 않는다" 항목이 그대로 있음을 `ls -ld` 로 재확인(특히 `~/dsv41-prep/kvfs`, `~/sglang-cmp/results/*.jsonl`, `vllm-dspark-fork`).
- 골든룰(노드마다): `dsv41-serve` active, `~/dsv41-prep/dsv41-override.env` 0 bytes, `systemctl --user show dsv41-serve -p Environment` 비어 있음, 클럭 캡 1989 MHz, `git -C ~/vllm-dsv41 status --porcelain` 비어 있음(`.pytest_cache`, `__pycache__` 는 무시 대상이라 원래 안 뜸), `:8888` health 200, `dsv41-watchdog.timer` active.
    - head 에서 `dsv41_ctl.sh status/caps` 를 쓸 때는 `DSV41_LOCAL_HOST=gx10-6040` 을 준다.
- 코드 변경 0: `git diff --stat` 이 `.notes/`, `.gitignore`, 두 README 만 건드림.
- `kvfs_verify.py` 는 돌리지 않는다(지운 것이 KV 저장소와 무관).

## 6. 이번에 하지 않는 것

오해 소지가 있는 것만 적는다.

- **서버 재기동, 벤치 재측정, 엔진 코드 변경 없음.** 채택 노브는 이미 `deploy/gb10-cluster/dsv41/dsv41.env` 기본값이고(36 `SHORT_RESERVE=4096`, 61 `SPEC_BLOCK_DROP=0`, 76 `DSV41_INDEXER_TP_SPLIT=1`, 118 `ENGRAM_RELEASE=0`, 121 `ENGRAM_PREFETCH=1`, 124 `ENGRAM_DECODE_ASYNC=1`, 132 `ENGRAM_CHUNK_RUNS=8`), 새 기준선 스위트는 #46 `prod-final` 로 끝났다.
- **노드 README 헝크 sync 와 main push** 는 #35 마무리에서 한다(오너 규칙).
- **옛 kvfs 저장소 37 GiB 삭제**는 오너 결정 대기라 손대지 않는다.
- **메모리 파일 갱신**은 매니저가 이슈를 닫을 때 한다.
- 로컬 homelab 클론과 노드 트리는 건드리지 않는다.

## 7. 화면이 바뀌는 것

사람이 보는 산출물 두 가지가 바뀐다.

- **hangar 보고서**: 같은 URL `https://hangar.tail39057.ts.net/p/dsv41-sglang-vs-vllm-2026-09-14/` 가 v5 로 교체된다. 제목·절 구성·스타일은 그대로고, 맨 위에 "v5 정정 (2026-09-16)" 절이 새로 붙는다. 표 1 의 마지막에서 두 번째 열 이름이 "우리 포크 기대치" 에서 `prod-final` 실측으로 바뀌고 값이 추정에서 실측으로 교체된다. 표 2 가 사라지고 한 문장으로 대체된다. "개선 계획" 표는 한 행(T2W Engram 준비 단계 이식)이 빠지고 기대치 열이 #39 몫 표로 바뀐다. 본문 여섯 군데 진단 문장이 새 귀속으로 교체된다.
- **README 두 파일**: 실측 표에 09-15~16 행이 한 줄 늘고, 포크 추가 기능 표의 세 행에 각각 한 구절이 붙는다.

## 8. 오너에게 물을 것

없다. 매니저 코멘트 `icmt-ef080914` 가 범위·삭제 목록·보존 목록·완료 기준을 다 정했고, 갈릴 수 있는 지점(옛 kvfs 저장소, 헤드 여유 레버 #31, 노드 sync 와 push 시점)은 이미 오너 결정으로 닫혀 있다. 계획 단계에서 새로 생긴 판단은 2-8 의 Grafana 각주 처리 하나이고, 이는 "인용 문장이 있으면" 이라는 조건이 성립하지 않아 에이전트가 정했다.

## 9. 작업 순서

1. `.gitignore` 에 `!.notes/2026-09-16-issue-42-baseline-report/` 추가.
2. v4 HTML 보존 → v5 작성 → `hangar upload -replace` → `hangar info` 로 v5 확인.
3. README 두 파일 갱신, 수치 대조.
4. 노드 삭제(head → f323 → 37cc → 27c4), 노드마다 확인/삭제/부재 확인을 `validation.md` 에 누적.
5. 네 노드 골든룰 재확인.
6. 커밋(`notes(dsv41): ...` + README), main 머지. push 는 하지 않는다.
