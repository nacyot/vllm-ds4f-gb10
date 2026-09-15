# 이슈 #36 계획 — 노브 조합 `combo-nob12x`(ENGRAM_RELEASE=0 + eagle block drop 끄기) 1회 실측 후 :8888 기본값 반영

작성 2026-09-15 KST, 워크트리 `issue-36-knob-combo`(base `main` dccd5b80f9). 매니저 레시피 icmt-ead4aa6b 를 따른다. 코드는 아직 건드리지 않았다. 조사는 전부 읽기 전용(로컬 파일, 헤드 `~/sglang-cmp/results/`, `dsv41-r0.log`, `systemctl is-active`).

## 1. 현재 동작과 문제점

### 1-1. 프로덕션 :8888 의 지금 상태 (읽기 전용 확인)

- 헤드 `~/vllm-dsv41` = 5374d4f481(main d369c6095d 의 배포 커밋, `git status` 깨끗). `dsv41-serve` active, 11:14:37 KST 부팅, health 200, 유휴 MemAvailable 5.84 GiB.
- 11:14 부팅 인자(r0.log `non-default args`): `mmap_release_after_steps=3`, speculative_config 에 `disable_eagle_block_drop` 없음, `moe_backend` 없음. 즉 **full-combo(10:43 부팅) 뒤 채택 기본값으로 이미 복구돼 있다.** 이번 작업의 시작점은 정상 프로덕션이다.

### 1-2. 세 노브의 현재 배선

| 노브 | 지금 | 소비처 |
| --- | --- | --- |
| `ENGRAM_RELEASE` | `dsv41.env:100` 기본 3 | `serve-node.sh:76` → `--engram-config {"mmap_release_after_steps":N}`; `dsv41_ctl.sh:15` KNOBS 목록에 있어 `--setenv` 로 4 노드에 전달됨 |
| eagle block drop 끄기 | 전용 노브 없음. `SPEC_EXTRA` 에 JSON 조각을 넣어야 하는데 `dsv41.env:53-54` 주석대로 `dsv41_ctl.sh --setenv` 를 지나며 따옴표가 깨진다(이전 실험은 이스케이프로 우회) | `serve-node.sh:106-108` 의 SC 문자열 → `--speculative-config` |
| `MOE_BACKEND=b12x` | `dsv41.env:61` 기본 비움 | `serve-node.sh:126`. **이번에 안 건드린다**(§5) |

### 1-3. 엔진 쪽 불변 조건 (엔진 코드 변경 없음, 확인만)

- `disable_eagle_block_drop` 은 `vllm/config/speculative.py:440` 의 SpeculativeConfig 필드다. `use_eagle_block_drop()`(:1886) 하나를 통해 세 소비처가 같은 값을 본다: 스케줄러(`v1/core/sched/scheduler.py:307,450` — 켜져 있으면 캐시 가능한 마지막 위치를 한 블록 뒤로 물림), `v1/core/kv_cache_utils.py:2120`, 오프로드 매니저(`v1/simple_kv_offload/manager.py:164`). 끄면 세 곳이 일관되게 뒤 블록을 유지한다. 부팅 시 한 번 읽는 값이라 런타임 토글이 아니다.
- 디스크 KV 티어의 저장소 정체성(`v1/kv_offload/file_mapper.py:96-127`: model_name, tokens_per_hash, blocks_per_file, TP/PP/PCP/DCP, dtype, KV 그룹 레이아웃)에 speculative/engram 설정은 **들어가지 않는다.** 따라서 두 노브를 바꿔도 `/mnt/kvdisk/kv/dsv41` 의 기존 세션은 그대로 재사용되고 config.json 불일치 거부도 없다.
- `ENGRAM_RELEASE=0` 은 `mmap_release_after_steps=0`: 프리페치 스레드가 N 스텝 전 프리폴트한 페이지를 `madvise` 로 내놓는 동작을 끄고 회수를 커널에 맡긴다(`dsv41.env:100` 주석). 페이지는 읽기 전용 파일 매핑이라 MemAvailable 계산에서 회수 가능으로 잡힌다. 단독 측정(fork-rel0, 03:22)에서 AG4 최저 6.81 GiB(시작 8.15), 콜드 32K +13.0%, 128K +10.4%.

### 1-4. 문제

full-combo(3 노브)는 속도 기준을 넉넉히 넘었지만 S128 중 헤드 MemAvailable 최저 2.66 GiB 로 기준(≥ 3.0) 미달이고, 그 원인은 b12x 의 헤드 비용(단독 b12x 도 2.96)이다. 나머지 두 노브의 조합 `combo-nob12x` 는 아직 측정된 적이 없고, block drop 끄기는 프로덕션에서 재현 가능한 노브가 없다.

## 2. 바꿀 것과 접근

### 2-1. 파일 변경 (deploy/ 만, 엔진 코드 0)

1. `deploy/gb10-cluster/dsv41/serve-node.sh` — 전용 노브 1개. `SPEC=dspark` 분기(:106-108)에서
   `[ "${SPEC_BLOCK_DROP:-1}" = "0" ] && SC="$SC,\"disable_eagle_block_drop\":true"` 를 `SPEC_EXTRA` 줄 앞에 추가. 1(기본)이면 SC 불변 → 기존 부팅 인자와 바이트 단위 동일.
2. `deploy/gb10-cluster/dsv41/dsv41_ctl.sh:15` — KNOBS 목록에 `SPEC_BLOCK_DROP` 추가(없으면 워크스테이션 환경 오버라이드가 노드에 안 간다).
3. `deploy/gb10-cluster/dsv41/dsv41.env` — `SPEC_REJECT` 아래에 `SPEC_BLOCK_DROP=${SPEC_BLOCK_DROP:-1}` 줄 + 주석(0 = `disable_eagle_block_drop`, 뒤 블록을 프리픽스 캐시에 남겨 후속 턴 TTFT 158K 3.13 → 1.17 s). `SPEC_EXTRA` 주석은 "전용 노브 예: SPEC_BLOCK_DROP" 로 손봄.
4. **판정 통과 뒤에만**: `dsv41.env` 기본값 `SPEC_BLOCK_DROP` 1 → 0, `ENGRAM_RELEASE` 3 → 0, 각 주석에 채택 날짜와 근거 수치(§3 표), README `Startup and cap checks` 의 "Production settings since 2026-09-13" 문단에 한 줄, `Operating rules` 의 복구 목록(`ENGRAM_PREFETCH=1, EMPTY_CACHE=1, EMPTY_CACHE_MIN_TOKENS=65536`)에 `ENGRAM_RELEASE=0, SPEC_BLOCK_DROP=0` 추가.
5. `.notes/2026-09-15-issue-36-knob-combo/results.md` — 대조표(fork-base2 / full-combo / combo-nob12x / 프로덕션 확인) + 판정 + 부팅·복구 기록.

커밋은 두 개로 나눈다: (a) 노브 배선(기본값 불변, 행동 불변), (b) 채택 기본값 + 문서. 실험 A 는 (a) 만 노드에 보낸 상태에서 **환경 오버라이드**로 켜므로 새 노브의 배선 자체가 실험으로 검증되고, 실패 시 (a) 만 남아도 프로덕션 동작은 그대로다.

### 2-2. 실험 A `combo-nob12x` — 부팅 1

전제: 다른 클라이언트 없음(직전 20 분 r0.log 의 `POST /v1` 출처가 127.0.0.1 뿐인지 확인), 캡 4 대 active 1989.

1. 로컬 커밋 (a) → `selftest_caps.sh`, `test_headroom.py` 통과 → `git format-patch` 를 4 노드 `~/vllm-dsv41` 에 `git am` (README 규칙).
2. `dsv41_ctl.sh stop` → `SPEC_BLOCK_DROP=0 ENGRAM_RELEASE=0 dsv41_ctl.sh start` (MOE_BACKEND 비움 = 기본). r0.log 부팅 인자에서 `'disable_eagle_block_drop': True` 와 `mmap_release_after_steps=0`, `moe_backend` 부재를 **로그로 확인**한 뒤에 측정한다.
3. health 200 → `dsv41_ctl.sh headroom` (레시피의 부팅 전 기록; 128K/158K 는 운영 문턱 256K 미만이라 게이트가 아니고 시작값 기록용) → 헤드 MemAvailable 기록.
4. 헤드 `~/sglang-cmp` 에서 `ab_suite.sh combo-nob12x http://127.0.0.1:8888` 의 셀을 **같은 명령·태그로 한 셀씩 전경 실행**(warm/warm2 → decodebench prose,code × c=1,4 → S8 → S32 → S128 → AG4r1), 이어 run_suite.sh 의 뒤 두 셀 `--mode hol 128000` 과 `longctx.py --records 10700`. 각 셀 사이에 stamp(시각·MemAvailable·클럭)를 `results/combo-nob12x.log` 에 남긴다. 래퍼를 통째로 돌리지 않는 이유: 한 명령이 10 분을 넘겨 전경 한도를 넘기고, "작은 전경 단계" 규칙에도 맞지 않는다. casebench/decodebench 는 같은 `results/*.jsonl` 에 append 되므로 결과는 래퍼 실행과 동일하게 비교된다.
5. 매 셀 뒤 `head_mem.min_gib` 와 stamp 를 보고 2.9 GiB 아래로 내려가면 중단(earlyoom 2.43).

기대(매니저): AG4 ≈ 216~221 s, S128 ≈ 1,500 tok/s, 최저 여유 ≈ 3.5 GiB.

### 2-3. 판정

세 기준 전부: AG4 총시간 214 s 근처(fork-base2 235.6 대비 명백한 개선이면 통과로 본다), 헤드 MemAvailable 최저 ≥ 3.0 GiB(모든 셀), 158K 니들 정답. 1 회 측정이며 매니저 레시피가 채택 구성의 확인 측정을 부팅 2 뒤의 짧은 확인(디코드 c=1/4 + S32)으로 정했으므로 그것을 따른다. 전체 스위트 재측정은 #42 기준선에서.

- 통과 → §2-4.
- 미달 → 기본값 변경 없이 `dsv41_ctl.sh start`(오버라이드 없음)로 복구, results.md 에 수치와 어느 기준이 왜 미달인지 기록, 이슈 코멘트. 어떤 노브를 빼고 갈지는 매니저/오너 판단(§4).

### 2-4. 프로덕션 반영 — 부팅 2

1. 로컬 커밋 (b) → selftest/test_headroom 재통과 → 4 노드 `git am`.
2. `dsv41_ctl.sh stop` → `dsv41_ctl.sh start`(환경 오버라이드 **없음**, 기본값만) → r0.log 부팅 인자로 두 값 확인 → health 200 → `caps` 4 대 active 1989 → 헤드 MemAvailable 기록.
3. 짧은 확인: decodebench prose,code × c=1,4 + S32(태그 `prod-nob12x`). 채택 수치의 ±10% 안이면 종료. 벗어나면 기록하고 보고(되돌리지 않는다: 1 회 잡음인지 판단은 매니저).
4. 골든룰 마감: health 200, caps, MemAvailable 을 results.md 와 이슈 코멘트에.

## 3. 영향 사이트 (호출자·소비자)

| 대상 | 영향 | 대응 |
| --- | --- | --- |
| `serve-node.sh` SC 빌더 | `SPEC=dspark` 에서만 한 조각 추가; `SPEC=none` 은 무관 | 기본 1 이면 문자열 불변 |
| `dsv41_ctl.sh` KNOBS | 목록 누락 시 오버라이드가 노드에 안 감 | 추가; `selftest_caps.sh` 는 SSH 스텁으로 호출만 기록하므로 통과 유지 |
| `dsv41.env` 소비자 = README 두 절, `.notes` | 채택 기본값 나열이 어긋남 | §2-1 4 번 |
| 프리픽스 캐시·오프로드 | 뒤 블록이 캐시에 남아 세션당 블록 수 +1(64 토큰), 후속 턴 히트가 늘어남. GPU KV 16 GiB / 호스트 2 GiB / 디스크 티어 예산은 그대로 | 저장소 정체성 불변(§1-3) → 기존 세션 재사용 |
| 수락 길이 | full-combo 코드 5.82 = 기준과 동일 → 분포 불변(block drop 은 캐시 정책) | 니들 정답으로 재확인 |
| 헤드 메모리 | `ENGRAM_RELEASE=0` 은 회수를 커널에 맡김; 단독 측정 최저 6.81(시작 8.15). 이번 신규 부팅은 #30 편차(4.6~7.1) 안에서 시작 | 모든 셀 최저 ≥ 3.0 을 기준으로 잡음 |
| 다른 실험·이슈 | #42 새 기준선은 이 기본값 위에서 잰다 | results.md 에 채택 커밋 해시 명시 |

## 4. 오너에게 물을 것 (plan-approve 에서)

없음. b12x 보류(#31 레버와 함께), EP=1 교환 안 함(#16 보류)은 이미 결정돼 있어 되묻지 않는다. 판정 미달 시의 부분 채택(둘 중 하나만) 여부만 그때 매니저에게 수치와 함께 올린다.

## 5. 이번에 하지 않을 것

- `MOE_BACKEND=b12x`, `EP=1`, `LINEAR_BACKEND` 어느 것도 켜지 않는다.
- 전체 스위트(run_suite.sh 12 셀) 재측정 없음 — 실험 A 는 ab_suite 셀 + HOL + longctx, 프로덕션 확인은 디코드 파동 + S32 뿐.
- 엔진 코드(`vllm/`) 변경 없음, `.so` 재빌드 없음(Bash/문서만이라 `git am` 으로 충분).
- 클럭·persistence·earlyoom 조작 없음, `rm` 없음(stop/start 의 shm 정리 함수만), 백그라운드 체인 없음.
- full-combo 중간 결과의 재해석은 매니저 표(§3 대조표에 인용)로 갈음하고 다시 돌리지 않는다.

## 6. 검증 방법 요약

- 로컬: `bash deploy/gb10-cluster/dsv41/selftest_caps.sh`, `.venv/bin/python -m pytest deploy/gb10-cluster/dsv41/test_headroom.py -q`, `bash -n serve-node.sh`, 그리고 `SPEC_BLOCK_DROP=0` / 미설정 두 경우의 SC 문자열을 `bash -x` 로 눈으로 대조.
- 노드: 부팅 인자 로그(두 값), health 200, caps, MemAvailable; 실험 A 세 기준; 부팅 2 짧은 확인; 실험 구간 `POST /v1` 출처 127.0.0.1 뿐임을 사후 grep.
