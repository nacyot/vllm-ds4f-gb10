# 이슈 #36 결과 — `combo-nob12x`(ENGRAM_RELEASE=0 + SPEC_BLOCK_DROP=0) 1회 실측 통과, :8888 기본값 반영·복구 완료

2026-09-15 14:39~15:02 KST, 워크트리 `issue-36-knob-combo`(base `main` dccd5b80f9). 계획 `plan.md` 의 부팅 두 번을 그대로 수행했다. 엔진 코드 변경 없음, `deploy/gb10-cluster/dsv41/` 만.

## 1. 결론

- **실험 A `combo-nob12x` 는 세 기준을 전부 통과**: 에이전트 4×6 총시간 208.0 s(기준 214 근처, fork-base2 235.6), 모든 셀 헤드 MemAvailable 최저 3.67 GiB(S128, 기준 ≥ 3.0), 158K 니들 정답(후속 턴 TTFT 0.77 s). 수락 길이 코드 5.82 불변.
- **채택**: `dsv41.env` 기본값 `SPEC_BLOCK_DROP=0`(신규 노브, `disable_eagle_block_drop`), `ENGRAM_RELEASE` 3 → 0. 프로덕션 :8888 은 14:57 오버라이드 없이 재부팅해 15:02 기준 health 200, 캡 4 대 active 1989 MHz(부하 중 측정), 헤드 MemAvailable 4.36 GiB.
- b12x 는 계획대로 켜지 않았다(full-combo 최저 2.66 GiB). EP=1 교환 없음.

## 2. 대조표 (전부 헤드 `~/sglang-cmp/results/*.jsonl`, 같은 도구·인자)

| 셀 | fork-base2 (9/13 채택) | fork-rel0 (release 0 단독) | full-combo (3노브) | **combo-nob12x (실험 A)** | prod-nob12x (부팅 2 확인) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 산문 디코드 c1 tok/s | 40.08 | 39.47 | 36.75 | 36.72 | 40.88 |
| 산문 디코드 c4 합산 | 81.27 | 86.91 | 89.85 | 85.51 | 85.49 |
| 코드 디코드 c1 | 78.73 | 82.31 | 88.35 | 81.34 | 83.71 |
| 코드 디코드 c4 합산 | 181.8 | 186.87 | 219.14 | 208.63 | 185.98 |
| 수락 길이(코드 tokens_per_chunk) | 5.82 | 5.82 | 5.82 | 5.82 | 5.82 |
| 콜드 8K tok/s | 1,147.8 | 1,818.7 | 1,849.3 | 1,770.9 | — |
| 콜드 32K tok/s | 1,459.2 | 1,649.3 | 1,972.9 | 1,917.8 | 1,915.3 |
| 콜드 128K tok/s | 1,362.7 | 1,504.6 | 1,294.0 | 1,464.4 | — |
| 에이전트 4×6 총시간 s | 235.6 | 220.7 | 181.7 / 176.4 | **208.0** | — |
| 에이전트 TTFT p50 s / 레인 디코드 tok/s | 5.33 / 19.2 | 3.91 / 20.5 | 2.96 / 25.2 | 3.30 / 21.3 | — |
| HOL 128K 중 짧은 요청 s | 88.9(보고서) | — | 81.0 | 70.77 | — |
| 158K 니들 / 후속 턴 TTFT s | 정답 / 3.13 | — | 정답 / 1.007 | **정답 / 0.77** | — |
| 헤드 MemAvailable 최저 GiB(셀) | 4.33 (S8) | 4.03 (S128) | 2.66 (S128) | **3.67 (S128)** | 4.32 (S32) |

실험 A 헤드 여유 경로(stamp): 시작 5.90 → 웜업 4.81 → S32 최저 4.38 → S128 최저 3.67, 종료 5.11 → AG4 최저 4.77 → HOL 뒤 8.53(웜업 컴팩션의 anon swap-out, #30 의 7.1 급 케이스) → longctx 뒤 9.16. 부팅 시작 여유 5.83(`headroom` 5.2 통과). 클럭 전 stamp 1989. 실험 구간(14:42~14:54) `POST /v1` 47 건 전부 127.0.0.1(외부 클라이언트 0; 직전 외부 요청 192.168.1.222 는 14:14).

부팅 2 확인은 디코드 c1/c4 + S32 만이다. 코드 c4 합산 186.0 은 실험 A 208.6 보다 11% 낮고 fork-rel0 186.9·기준 181.8 과 같은 수준이다. 이 셀은 구성별 1회값이 166.6(hit-eagle)~208.6 으로 흔들리는 셀이라 잡음으로 보되, #42 새 기준선에서 재확인한다. 나머지(산문 c1/c4, 코드 c1, S32)는 실험 A 의 ±5% 안이다.

## 3. 변경 (커밋)

- fcfd1bf086 `deploy(dsv41): SPEC_BLOCK_DROP knob toggles disable_eagle_block_drop` — serve-node.sh SC 빌더에 한 줄, dsv41_ctl.sh KNOBS 목록, dsv41.env 항목(기본 1 = 행동 불변). 실험 A 는 이 커밋만 노드에 두고 `SPEC_BLOCK_DROP=0 ENGRAM_RELEASE=0 dsv41_ctl.sh start` 로 부팅했으므로 노브 배선이 실험으로 검증됐다(r0.log 부팅 인자 `'disable_eagle_block_drop': True`, `mmap_release_after_steps=0`, `moe_backend` 없음).
- eb791041eb `deploy(dsv41): adopt SPEC_BLOCK_DROP=0 and ENGRAM_RELEASE=0 as the defaults` — dsv41.env 기본값·주석, README(2026-09-15 문단 + Operating rules 복구 목록).
- 로컬 검증: `selftest_caps.sh` 전부 PASS(exit 0), `test_headroom.py` 13 passed, `bash -n` 두 스크립트, SC 문자열 세 경우(미설정/0/1) 대조.

## 4. 노드 반영 기록

- 4 노드 `~/vllm-dsv41` 에 `git format-patch | git am`. 워커 3 대는 커미터 identity 가 없어 `git -c user.name=nacyot -c user.email=<head 저장소 값> am` 으로 적용(첫 시도 실패분은 `git am --abort` 뒤 재적용, 헤드는 git worktree 라 rebase-apply 가 `.git/worktrees/vllm-dsv41/` 아래에 있었다).
- 두 번째 패치의 README hunk 는 노드에서 적용되지 않았다: 노드 deploy README 는 401413ceea 시점(9/13 이전, vision·kvfs-gc·spec 문단 없음)이라 main 과 이미 여러 문서 커밋만큼 벌어져 있다. 문서만 다르므로 `--exclude=deploy/gb10-cluster/dsv41/README.md` 로 env 만 적용했다. 헤드 cf9c0a7198, f323 37de7eaf5, 37cc 3c720c969, 27c4 a0d0e6b31, 전부 dirty 0. **노드 README 동기화는 이번 범위 밖**(별도 문서 sync 때 한 번에).
- 부팅 2: 14:56:21 stop → 14:57:18 start(오버라이드 0, unit Environment 에 SPEC/ENGRAM 없음) → 14:59:34 health 200(130 s) → r0.log 부팅 인자 두 값 확인 → 헤드 5.83 GiB.

## 5. 골든룰 마감 (15:02 KST)

health 200, `dsv41_ctl.sh caps` 4 대 active/Enabled/1989(디코드 요청 중 측정; 유휴 직후 측정은 208~305 MHz 로 읽히니 부하 중에 볼 것), 헤드 MemAvailable 4.36 GiB(짧은 프로브 뒤 정상 범위). 클럭·persistence 조작 없음, `rm` 없음(stop/start 의 shm 정리 함수만: `/dev/shm/vllm_offload_*.mmap` 1 건씩).

## 6. 남는 것

- 전체 스위트 재측정은 #42 의 새 기준선에서(이 기본값 위). 코드 c4 디코드 셀의 잡음 폭(§2)도 그때 본다.
- b12x 는 헤드 여유 레버(#31, 오너 결정)와 함께 다시 본다.
- 노드 deploy README 가 main 보다 뒤처져 있다(문서만). 다음 문서 sync 때 정리.
