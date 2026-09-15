# 이슈 #38 결과 — 인덱서 프리필 TP 분할(`DSV41_INDEXER_TP_SPLIT`) 이식과 꺼짐/켜짐 실측

작성 2026-09-15 KST, 워크트리 `issue-38-indexer-tp-split`(base `main` c8cb661ceb). 계획은 같은 디렉터리 `plan.md`.

## 1. 구현

- `25441ae260` sparse_attn_indexer: T2W 패치(t2w-best `git diff`, 베이스 e47aa780bc)에서 SM12x hunk 를 뺀 4개 hunk 를 로직 그대로 적용했다. 우리 트리는 이미 SM12x hunk 를 가지고 있다. 배너 주석 15줄은 5줄로 줄였고, 출처 주석 2줄을 지웠다. ruff format 이 T2W 의 여러 줄 호출 몇 개를 한 줄로 합쳤다(행동 불변).
- 같은 커밋의 `tests/v1/attention/test_indexer_tp_split.py`: max10 초안 3개에 `test_env_mismatch_refuses_start` 를 더했다. 초안의 `calls` 에는 mypy 가 요구한 타입 주석을 붙였다.
- `e0e57b8383` deploy: `dsv41.env` 에 `DSV41_INDEXER_TP_SPLIT=${…:-0}` 을 넣었다. `serve-node.sh` 의 `set -a` 가 export 하므로 그 파일은 고치지 않았다. `dsv41_ctl.sh` KNOBS 에 이름을 추가했다.
- 로컬: pre-commit(ruff check/format, typos, shellcheck, mypy 3.10) 통과. `bash -n` 세 파일 통과, `selftest_caps.sh` 전부 PASS.

## 2. 노드 반영과 단위 테스트

- 16:39:27 프로덕션 정지. 정지 직전 요청은 running 0, waiting 0 이었고 로그에는 Grafana `/metrics` 수집만 있었다. 헤드 headroom 8.53 GiB 였다.
- 정지 전에 노드 4대에서 `git apply --check` 가 통과했고, 정지 뒤 `git -c user.name -c user.email am` 으로 두 커밋을 적용했다. 4대 모두 dirty 0 이다(노드마다 커밋 SHA 는 다르고, 내용은 로컬 `25441ae260`, `e0e57b8383` 과 같다).
- 헤드(서버 정지 상태): `pytest tests/v1/attention/test_indexer_tp_split.py` **30 passed**(7.0 s, CPU).

## 3. 랭크 간 env 불일치 시 기동 거부 (부팅 M)

- 16:40:49 워커 gx10-27c4 한 대만 `systemctl --user set-environment DSV41_INDEXER_TP_SPLIT=1`, 16:40:58 기본값으로 `start`.
- 16:41:56(시작 후 58 s, 가중치 로드 전) **4랭크 모두** 같은 줄로 죽었다:
  `RuntimeError: DSV41_INDEXER_TP_SPLIT / DSV41_INDEXER_TP_SPLIT_MIN differ across TP ranks: [(False, 512), (False, 512), (False, 512), (True, 512)]. Set them identically on every node.`
- 16:42:18 4대 unit inactive, 교착 없음. `stop` 으로 정리했고, 16:42:40 `unset-environment` 뒤 4대 모두 `show-environment` 의 DSV41 항목 0 을 확인했다.

## 4. 부팅 A `i38-off` (분할 꺼짐, main 기본값)

- 16:42:48 `start`(오버라이드 없음), 16:45:28 health 200. r0.log 의 분할 로그 0 줄, `mmap_decode_async=1, mmap_min_chunk_runs=8`, 헤드 5.86 GiB.
- 웜업은 `run_suite.sh` 와 같게 solo 16K 와 mixed 6K+2 를 돌렸다. 이어서 casebench solo 콜드 프리필(요청마다 새 내용)을 돌렸다.

| 태그 | r1 | r2 | r3 | 헤드 최저 GiB |
| --- | ---: | ---: | ---: | ---: |
| S32 tok/s | 1,535 | 1,953 | 1,778 | 6.10 / 6.74 / 8.52 |
| S128 tok/s | 1,382 | 1,645 | 1,687 | 6.62 / 7.52 / 8.06 |

부팅 뒤 첫 긴 프리필은 느리다(`dsv41.env` 로더 주석의 "first long cold prefill after a boot runs 2-15 s slower"). 그래서 비교는 r2·r3 로 한다. S32 는 r2·r3 사이도 9% 흔들려 방향 확인용으로만 보고, 판정은 r2·r3 가 2.5% 안으로 모인 S128 로 한다.

- decodebench c1 gen 256: 산문 `tokens_per_chunk` **3.12**(스트림당 45.4 tok/s), 코드 **5.82**(84.5 tok/s).
- greedy 대조군(`greedy_probe.py.txt`, 같은 프롬프트 4회, 매번 새 `cache_salt`)에서 **분할 꺼짐끼리도 출력이 비트 단위로 같지 않았다**:
    - 8K(프롬프트 10,459 토큰): 첫 토큰이 near-tie 다. `station` −0.317 / `Record` −1.317 과 `Record` −0.477 / `station` −0.977 이 회차마다 뒤바뀌어 텍스트가 4회 중 3가지로 갈렸다. 이 프롬프트는 동일성 판정에 쓸 수 없고, 첫 토큰 logprob 분포 비교에만 쓴다.
    - 32K(42,970 토큰): 첫 토큰 `station` −0.012 ~ −0.002(차순위와 4.5 nats 이상 차이). 4회 중 3회는 텍스트가 같았고, 1회는 줄 끝에 `units` 를 붙이는 쪽으로 9번째 토큰에서 갈렸다. 6쌍의 공통 접두 top-1 logprob 차는 0.13~0.35 nats.
    - 계획 §7-1 의 경우(대조군부터 흔들림)에 해당한다. 판정은 권고안대로 켜짐의 차이가 이 대조군 범위 안인지로 한다.

## 5. 부팅 B `i38-on` (`DSV41_INDEXER_TP_SPLIT=1`)

- 16:55:37 `DSV41_INDEXER_TP_SPLIT=1 dsv41_ctl.sh start`(KNOBS 로 4대 전달), 16:58:21 health 200.
- 4랭크 로그에 `DSV41_INDEXER_TP_SPLIT on: indexer prefill rows split across 4 TP ranks for chunks with >= 512 rows.` 가 1줄씩 있다(16:56:32).
- 웜업은 A 와 같다. 이어서 S8 1회, S32 ×3, S128 ×3 을 돌렸다.

| 태그 | r1 | r2 | r3 | 헤드 최저 GiB |
| --- | ---: | ---: | ---: | ---: |
| S8 tok/s | 1,750 | | | 5.64 |
| S32 tok/s | 1,966 | 1,990 | 1,967 | 5.61 / 5.60 / 5.57 |
| S128 tok/s | 1,726 | 1,828 | 1,844 | 5.04 / 5.12 / 5.20 |

### 꺼짐 대비

| | 꺼짐 r2·r3 평균 | 켜짐 r2·r3 평균 | 차이 |
| --- | ---: | ---: | ---: |
| S32 | 1,866 (±4.7%) | 1,978 (±0.6%) | +6.0% |
| S128 | 1,666 (±1.3%) | 1,836 (±0.4%) | **+10.2%** |

- S128 은 켜짐의 가장 느린 r1(1,726)도 꺼짐의 가장 빠른 회차(1,687)보다 빠르다. 이득은 잡음 밖이다. 완료 기준 128K ≥ 1,700 tok/s 도 세 회차 모두 넘었다.
- S32 이득이 S128 보다 작은 것은 매니저 §1 의 예상과 같다. 점수 비용이 누적 K 에 비례해 긴 컨텍스트에서 몫이 커진다. 켜짐 쪽 S32 는 첫 회차 저하도 없었다. 꺼짐의 r1 저하가 부팅 뒤 첫 긴 프리필 탓인지 인덱서 몫인지는 이 데이터로 가를 수 없어 판정에 쓰지 않는다.
- 헤드 MemAvailable 최저 5.04 GiB(판정선 3.0 이상). 켜짐 부팅의 시작값이 5.6 GiB 로 A(6.1~9.6)보다 낮았던 것은 부팅마다 다른 웜업 컴팩션 편차다(RC#16 #30: 신규 부팅 여유 4.6~7.8 편차).

### 디코드와 158K 니들

- decodebench c1 gen 256: 산문 `tokens_per_chunk` **3.12**(46.0 tok/s), 코드 **5.82**(85.9 tok/s). 꺼짐과 수락 길이가 같다. 매니저 §2 가 경고한 t2w-best 의 산문 2.44 는 분할에서 온 것이 아니다.
- `longctx.py --records 10700`: 158,515 토큰 콜드 TTFT 77.9 s(**2,034 tok/s**, prod-i37 1,880, combo-nob12x 1,568), 답 `12` 정답. 후속 턴 TTFT 0.67 s(prod-i37 0.75), 답 `986` 정답.

### greedy 출력 비교 (부팅당 4회, 같은 프롬프트)

첫 토큰:

| | 8K 첫 토큰 (top-1 logprob) | 32K `station` logprob / 차순위 `Record` |
| --- | --- | --- |
| 꺼짐 | station −0.317, Record −0.477, Record −0.256, station −0.479 | −0.012/−4.51, −0.007/−5.26, −0.003/−6.00, −0.002/−6.25 |
| 켜짐 | station −0.391, −0.257, −0.580, −0.319 | −0.013/−4.51, −0.040/−3.29, −0.042/−3.29, −0.001/−6.75 |

쌍별 비교(`greedy_probe.py compare`, 공통 접두 top-1 logprob 차):

| 크기 | 묶음 | 쌍 | 텍스트 동일 | 첫 불일치 토큰 | 차 중앙값/최대 |
| --- | --- | ---: | ---: | --- | --- |
| 8K | 꺼짐-꺼짐 | 6 | 1 | 0,0,0,0,5 | 0.000 / 0.221 |
| 8K | 꺼짐-켜짐 | 16 | 3 | 0×8, 5×4, 9 | 0.001 / 0.263 |
| 8K | 켜짐-켜짐 | 6 | 3 | 5,5,5 | 0.189 / 0.323 |
| 32K | 꺼짐-꺼짐 | 6 | 3 | 9,9,9 | 0.223 / 0.349 |
| 32K | 꺼짐-켜짐 | 16 | 9 | 5×4, 9×3 | 0.188 / 0.492 |
| 32K | 켜짐-켜짐 | 6 | 3 | 5,5,5 | 0.235 / 0.492 |

- 32K 첫 토큰의 argmax 는 8회 모두 `station` 이다. 꺼짐-켜짐 쌍의 텍스트 동일 비율(9/16)은 꺼짐-꺼짐(3/6)보다 낮지 않다.
- 켜짐 쪽 한 회차(`i38-on-b#0`)가 5번째 토큰에서 `logged … units at` 변형으로 갈렸고, 두 회차가 차순위 −3.29 상태를 보였다. 그래서 꺼짐-켜짐 최대 차(0.49)와 첫 불일치(5)가 꺼짐-꺼짐 범위(0.35, 9)를 벗어났다. 다만 이 두 극값은 켜짐-켜짐 안에도 그대로 있다. 계획 §5-2 의 기준으로는 "켜짐만 더 크게 갈라짐"과 "표본 4개의 대조군 범위가 좁음"을 가를 수 없다.
- 판정을 가르기 위해 서버를 내린 창에서 커널 수준 동일성을 직접 확인한다(§6).
- 17:10:41 부팅 B `stop`(직전 running 0, waiting 0, shm 정리 깨끗).

## 6. 커널 수준 동일성 (헤드 GPU, 서버 정지 상태)

`split_equivalence.py.txt`: 프로덕션 FP8 인덱서 캐시 배치(`use_fp4_cache=False`, `q_scale` 없음, 헤드 32 × 128, topk 512, 후보 2048 × 8)에 합성 입력을 넣었다. 실제 커널(DeepGEMM `fp8_fp4_mqa_logits`, `top_k_per_row_prefill`, 후보 select/mask Triton)로 비분할 루프 본문과 `_dsv41_prefill_rows_local` 4랭크 조각 + 순서대로 모으기(all-gather 흉내)를 비교했다. 청크 5종: 4096 행 × 124K 문맥, 28K 문맥, 첫 청크, 1000 행(패딩 경로), 압축비 2.

1차 실행(17:12):

- **logits: 5종 모두 비트 동일**(행별 유효 구간 [ks, ke) 비교, 최대 차 0). 커널이 행 수에 따라 다른 값을 내지 않는다.
- **후보 블록(20 층 `candidate_write`): 5종 모두 비트 동일.**
- top-k 인덱스 텐서는 분할-비분할도 달랐지만, **비분할을 두 번 돌린 것끼리도 달랐다.** `top_k_per_row_prefill` 은 같은 입력에서도 출력 순서·동점 선택이 매번 같지 않다. 텐서 동일 비교로는 분할 여부를 가를 수 없어, 행별 인덱스 집합과 선택된 점수 다중집합으로 다시 비교한다(2차).

2차 실행(17:13):

- 5종 모두 **행별 인덱스 집합이 다른 행 0 개, 선택된 점수 다중집합이 다른 행 0 개**다. 후보 원천 층, 후보 마스크 층, 후보 없는 층 전부 같았고, 비분할 반복끼리도 같았다.
- 1차의 텐서 불일치는 한 행 안의 **출력 순서**뿐이다. 이 순서는 분할 없이 같은 입력을 두 번 돌려도 바뀐다.

결론: 분할 경로는 비분할 루프와 **같은 logits(비트 동일), 같은 후보 블록(비트 동일), 행마다 같은 top-k 인덱스 집합**을 만든다. §5 의 greedy 텍스트 흔들림은 분할과 무관한 기존 재계산 비결정성이다(행 안 인덱스 순서가 매번 달라 뒤의 sparse attention 합산 순서가 바뀌는 등). 분할 켜짐 쪽에서 본 극값(차 0.49, 5번째 토큰 분기)은 표본 4개 대조군의 범위 추정 한계로 본다.

## 7. 판정: 채택 (`DSV41_INDEXER_TP_SPLIT=1` 기본값)

| 계획 §5-4 조건 | 결과 |
| --- | --- |
| 1. 출력 동일성 | 커널 수준 동일(§6). 텍스트 흔들림은 꺼짐 대조군에도 있는 기존 비결정성(§4, §5), 계획 §7-1 권고안 기준 통과 |
| 2. 수락 길이 | 산문 3.12 / 코드 5.82, 꺼짐과 동일 |
| 3. 158K 니들 | 정답, 후속 턴 정답, 콜드 2,034 tok/s, 후속 TTFT 0.67 s |
| 4. 잡음 밖 이득 | S128 +10.2%(1,666 → 1,836, r2·r3), S32 +6.0%. 128K ≥ 1,700 완료 기준 충족 |
| 5. 헤드 최저 ≥ 3.0 GiB | 5.04 GiB |
| 6. 기동 거부 | 4랭크 `RuntimeError`, 가중치 로드 전, 교착 없음(§3) |

## 8. 채택 반영과 프로덕션 복구 (골든룰)

- `c273135f80` deploy: `dsv41.env` 기본값을 1 로 바꾸고 주석에 수치를 적었다. `README.md` 에 #38 문단을 넣고, 복구 기본값 목록에 `DSV41_INDEXER_TP_SPLIT=1` 을 더했다.
- 노드 4대에 `git am --exclude=deploy/gb10-cluster/dsv41/README.md` 로 적용했다. 4대 모두 dirty 0 이고 `DSV41_INDEXER_TP_SPLIT:-1` 을 확인했다.
- 17:14:44 오버라이드 없이(로컬 knob env 0개) `start`, 17:17:37 health 200. 이번 부팅의 4랭크 로그에 `DSV41_INDEXER_TP_SPLIT on` 이 1줄씩 있다.
- 웜업 solo 16K 1,759 tok/s(헤드 최저 4.66 GiB).
- decodebench c1 gen 256(`prod-i38`): 산문 `tokens_per_chunk` 2.94(43.2 tok/s), 코드 5.82(85.4 tok/s). 산문은 기존 프로덕션 범위(2.91~3.08) 안이다.
- 17:18:22 `dsv41_ctl.sh caps`: 4대 모두 `gpu-clock-cap.service` active, persistence Enabled, MAX_SM 1989 MHz. 디코드 측정 직후에 찍혔지만, 이 명령은 설정된 최대 클럭을 읽으므로 부하 여부와 무관하다.
- 헤드 MemAvailable 4.76 GiB, health 200.
- 프로덕션 다운타임: 16:39:27 ~ 16:45:28(M, A 부팅), 16:54:42 ~ 16:58:21, 17:10:41 ~ 17:17:37. A·B 부팅 중에도 :8888 은 떠 있었다(측정 전용, 외부 요청 없음).
- 노드에 남은 작업물은 `~/dsv41-prep/i38/`(프로브, `greedy.jsonl`, `greedy_compare.jsonl`), `~/dsv41-prep/i38-a/`, `~/dsv41-prep/i38-b/`(패치)다. 벤치 기록은 `~/sglang-cmp/results/{casebench,decode,longctx}.jsonl` 의 `i38-off*`, `i38-on*`, `prod-i38*` 태그다. 삭제는 하지 않았다.
