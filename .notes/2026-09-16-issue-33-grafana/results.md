# 이슈 #33 결과: 프리필 패널이 계산 토큰만 세고, 요청당 속도는 실측과 1.2% 차이다

측정 2026-09-16 01:37~01:47 KST, 프로덕션 :8888. 재기동과 설정 변경은 없었다. 워치독 타이머는 01:40:20 에 멈췄고 01:44:58 에 다시 켰다. 구현 기록은 `implement.md`, 계획은 `plan.md` 에 있다.

## 판정

| 완료 기준 | 결과 | 판정 |
| --- | --- | --- |
| 32K 3 회: 요청당 프리필 속도 패널(72, 5 분)이 실제 속도와 10% 안 | 패널 1,867.3 tok/s, casebench Σ95,826 토큰 / Σ51.93 s = 1,845.3 tok/s, +1.2% | 통과 |
| 32K 3 회: 계산 토큰 증가가 보낸 프롬프트 토큰과 맞음 | `increase(local_compute[5m])` 95,826 = Σprompt_tokens 95,826 | 통과 |
| 32K 3 회: 정지 식 0 | 01:40~01:43 12 샘플 모두 0 | 통과 |
| 493K 복원 1 회: 프리필 패널 0 근처 | `local_compute` rate 0~2 tok/s(복원 뒤 남은 계산 87 토큰) | 통과 |
| 493K 복원 1 회: 복원 패널이 로드 토큰을 보임 | `external_kv_transfer` +492,928, rate 8,215~16,430 tok/s | 통과 |
| 라이브 대시보드 반영 | 01:35:24 반영, 커밋본과 패널별 diff 0 | 통과 |
| vmalert 규칙 교체 | `ds4f.agent-stall` 3 규칙 health ok, lastError 0 | 통과 |
| 골든룰 | 아래 "종료 상태" | 통과 |

## 32K 프리필 3 회

명령은 헤드에서 `python3 ~/sglang-cmp/casebench.py --mode solo --prefills 1 --prefill-tokens 32000 --config prod-i33 --tag i33-s32-r{1,2,3}` 이다. 프롬프트는 회차마다 새로 만들어 캐시에 맞지 않는다(`local_cache_hit` 증가 0). 카운터는 헤드 `/metrics` 에서 회차 전후로 직접 읽었다.

| 회차 | 시작 | 프롬프트 토큰 | casebench wall | casebench tok/s | `local_compute` 증가 | `prefill_time_sum` 증가 | 카운터 기준 tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | 01:40:20 | 31,902 | 18.36 s | 1,737.8 | 31,902 | 18.152 s | 1,757.5 |
| r2 | 01:41:02 | 32,003 | 16.89 s | 1,894.9 | 32,003 | 16.713 s | 1,914.9 |
| r3 | 01:41:43 | 31,921 | 16.68 s | 1,914.0 | 31,921 | 16.454 s | 1,940.0 |
| 합 | | 95,826 | 51.93 s | 1,845.3 | 95,826 | 51.318 s | 1,867.3 |

`kv_computed_sum` 증가도 회차마다 `local_compute` 증가와 같았다. casebench wall 은 HTTP·토크나이즈를 포함하고 `prefill_time` 은 스케줄에서 첫 토큰까지라 카운터 쪽이 1% 남짓 빠르다.

### VM 에서 본 패널 값

72 패널의 5 분 창 끝을 01:43:30 으로 잡았다. 이 창에는 r1~r3 만 들어가고 01:38:15 에 끝난 외부 요청은 빠진다. 창 끝을 01:42:15~01:42:45 로 잡으면 그 외부 요청이 창에 들어가 성공 4 건, `local_compute` 95,831, 72 = 1,859.8 이 된다.

| 식 (T = 01:43:30, 5 분) | 값 |
| --- | ---: |
| 72: `increase(kv_computed_sum) / increase(prefill_time_sum)` | 1,867.3 tok/s |
| `increase(prompt_tokens_by_source{local_compute})` | 95,826 |
| `increase(request_prefill_kv_computed_tokens_sum)` | 95,826 |
| `increase(request_prefill_time_seconds_sum)` | 51.318 s |
| `increase(request_success_total)` | 3 |
| `rate(local_compute[5m])` | 319.4 tok/s (처리량, 속도 아님) |

1 분 rate 패널(3)은 r1~r3 동안 톱니로 1K tok/s 근처까지 올랐다가 0 으로 돌아왔다. 첫 토큰 시점에 몰아 기록되기 때문이다.

## 493K 복원 1 회

헤드 MemAvailable 은 8.59 GiB 로 5.2 GiB 게이트를 넘었다. 이 조건에서 09-12 에 복원에 쓰던 salt `P13`(33,000 records)을 다시 보냈다. fs 저장소는 09-11 부터 이름이 같은 디렉터리(`…_06d2a1a0be2b_r0`) 하나이고 오늘도 쓰이고 있어 복원될 것으로 봤다. 명령은 `~/vllm-dsv41/.venv/bin/python kvoff_probe.py i33-restore P13 33000`(레포 사본) 이다. 첫 시도는 `deploy/.../.venv/bin/python` 경로 오타로 요청 없이 끝났다(exit 127, 카운터 불변).

- 01:44:00 시작, TTFT 8.614 s, 프롬프트 493,015 토큰, 답 `12` 정답, MemAvailable 최저 7.89 GiB.
- probe 차분: `external_prefix_cache_hits` 492,928, `kv_offload_load_bytes` 4.45 GB, `prefix_cache_hits` 0.
- `/metrics` 차분: `local_compute` +87, `external_kv_transfer` +492,928, `kv_computed_sum` +87, `prefill_time_sum` +0.486 s, 성공 +1(stop).

VM 에서 본 복원 창(1 분 rate, 15 s 간격):

| 시각 | 43/3 새 프리필 식 | 40 복원 식 | 옛 프리필 식 |
| --- | ---: | ---: | ---: |
| 01:44:15 | 0 | 0 | 0 |
| 01:44:30 | 1 | 8,215 | 8,216 |
| 01:45:00 | 1 | 10,953 | 10,955 |
| 01:45:15 | 2 | 16,430 | 16,433 |
| 01:45:30 | 0 | 0 | 0 |

옛 식은 같은 복원을 프리필 8~16K tok/s 로 보였고, 새 식은 복원 뒤 남은 계산 토큰만 보인다.

### 계획 단계 미확인 항목의 답

- 복원 요청의 `request_prefill_kv_computed_tokens` 는 복원분을 빼고 센다(+87). 72 의 분자는 계획대로 둔다.
- 복원 대기 시간은 `request_prefill_time_seconds` 에 거의 들어가지 않는다(TTFT 8.6 s 중 0.49 s). 계획서와 첫 배포의 72 설명에는 "복원 요청은 복원 시간이 분모에 들어가 낮게 보인다" 고 적었는데 틀린 설명이었다. homelab `2eb9be6` 에서 "복원 대기는 큐 시간으로 잡혀 거의 섞이지 않는다(493K 복원 1회: +87 토큰, +0.49초)" 로 고쳤다. Forgejo `main` 에 fast-forward push 했고, Flux `apps` 가 `2eb9be6` 을 적용했다. 라이브 설명은 01:47:54 KST 에 바뀌었다.

## 관찰과 한계

- 정지 식은 01:44:15 한 샘플(15 s)에서 1 이었다. 복원 요청이 원격 KV 를 기다리는 동안 `waiting` 1, `running` 0 이었다. 출력 스텝도 없었다. 서버가 그 전 2 분 동안 유휴라 KV 사용률도 그대로였다. 유휴 서버에 온 첫 요청이 원격 KV 를 기다리는 몇 초 동안은 이 식이 1 이 될 수 있다. 옛 식도 같은 조건에서 1 이다. 한 샘플이고 알림 규칙에는 쓰이지 않아 식은 바꾸지 않았다.
- 7 일 창에서 정지로 뜬 분 수는 라이브 옛 식 3, 매니저 식 3, 새 식 2 였다(계획 단계 집계).
- `DS4FRunawayFinish` 는 규칙 재적재 직후 firing 이었다. 실제 트래픽에서 max_tokens 에 걸린 종료(length) 2 건이 원인이다. casebench 는 `max_tokens 1` 이라 회차마다 length 로 끝나 이 규칙을 다시 울릴 수 있다. 측정 부산물이고 규칙은 바꾸지 않았다.
- 헤드에서 `dsv41_ctl.sh status/caps` 를 그대로 돌리면 헤드가 자기 자신에게 ssh 하다 publickey 로 막혀 "cap released on gx10-6040" 이 뜬다. 캡은 해제되지 않았다. `DSV41_LOCAL_HOST=gx10-6040` 을 주면 로컬 프로브로 확인되고 exit 0 이다.

## 화면 확인

agent-browser(1600 폭)로 라이브 대시보드를 봤다.

- 제목은 `DSv4.1F vLLM (gx10)` 이다.
- 처리율 행은 "프리필 속도, 계산 토큰", "요청당 프리필 속도 (5분, 종료 요청 기준)", "디코드 속도" 세 칸이다.
- 요청당 속도 패널은 r1~r3 뒤 1.87K 로 올라섰다.
- 실시간 패널의 프리필 값은 복원 직후 1.45 tok/s 였다.
- 정지 시그니처 제목은 새 기준(대기≥1, 실행<16, 출력 스텝≈0, KV 변화 없음)을 보이고, 01:44 에 위 한 샘플만 1 이다.
- 에이전트 턴 정지 감시 행에서 빈 감지기 패널이 사라졌다. 툴콜 파서 결과가 전체 폭, stat 3 개가 1/3 폭씩이다.
- 시스템·GPU 행 제목은 "4 노드" 다.
- 복원 패널(40)은 캡처 범위(4000 px) 아래라 화면으로는 보지 않았고 위 VM 값으로 확인했다.

## 종료 상태 (01:46 KST)

- health 200, 4 노드 `dsv41-serve` active.
- override 0 bytes, `Environment` 비어 있음, `~/vllm-dsv41` 트리 clean(4 노드).
- `DSV41_LOCAL_HOST=gx10-6040 dsv41_ctl.sh caps` exit 0. 캡 서비스 4 대 active, 최대 SM 클럭은 6040/f323/37cc 1989 MHz, 27c4 1995 MHz(샘플 최대치, 게이트 통과).
- `dsv41-watchdog.timer` active, 재시작 뒤 첫 실행 결과 0, `dsv41-ATTENTION` 없음.
- 헤드 MemAvailable 8.33 GiB.
- 헤드 `~/vmagent/scrape.yml` 은 두었다(쓰이지 않는 파일).
