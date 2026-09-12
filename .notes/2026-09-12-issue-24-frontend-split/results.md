# 이슈 #24 결과 — 원격 프런트엔드 스파이크 (2026-09-12 20:11–21:03 KST)

워크트리 `issue-24-frontend-split`, 커밋 7d771bb2da → 589e5afc0d(런처 4개). 노드 4대 `~/vllm-dsv41` 에 `git am` 으로 동일 반영(dsv41_ctl.sh 는 워크스테이션 전용이라 노드에서 제외). 프런트엔드 gx10-f323(10.100.0.32:8889), head gx10-6040 `--headless`, RPC 29560.

## 결론

1. **된다.** DP=1·TP=4 에서 엔진 없는 원격 프런트엔드(`--data-parallel-size-local 0`) + headless head 엔진이 핸드셰이크·서빙한다. 8K 니들 정답, 8K/32K TTFT 운영과 동일(±1%).
2. **head 메모리 절감은 0.** headless head 에는 `vllm serve --headless` 부모 프로세스(`run_headless` → `CoreEngineProcManager`)가 남고 그 PSS 가 API 서버 프로세스와 같다(1,011 vs 1,015 MB). 유휴 MemAvailable 은 부팅 직후 5.67 vs 5.70 GiB, 8K/32K 프로브 후 3.96 vs 3.95 GiB 로 차이 없음. 프런트엔드를 받은 f323 은 0.83 GiB 를 더 쓴다.
3. **프런트엔드 단독 재시작 불가.** 재시작한 프런트엔드는 3분 동안 health 가 오지 않고(엔진 HELLO 대기) head 엔진은 재핸드셰이크 없이 그대로 남는다(로그 없음). 프런트엔드 사망도 감지하지 않는다(1차 부팅에서 프런트엔드 출력 핸들러가 죽은 뒤 엔진·워커 그대로 잔류).
4. 매니저 코멘트의 코드 가정 두 곳이 실기에서 틀렸고 런처로 보정했다(아래).

## 런처에서 보정한 것 (vLLM 파이썬 변경 없음)

- **DP=1 이면 `--data-parallel-address` 가 무시된다.** `ParallelConfig.__post_init__`(`vllm/config/parallel.py:899-925`)는 `data_parallel_size > 1 or data_parallel_size_local == 0` 일 때만 CLI 값을 쓰고, head(dp=1, local=1)는 else 로 가서 `VLLM_DP_MASTER_IP`(기본 127.0.0.1)로 덮는다. 1차 부팅 head 로그: `headless mode, with head node address tcp://127.0.0.1:29560`. → serve-node.sh headless 분기에서 `export VLLM_DP_MASTER_IP=$FRONTEND_ADDR`(5ccfbf1434). RPC 포트는 CLI 값이 유지된다.
- **프런트엔드에 엔진과 같은 `--kv-transfer-config` 가 필요하다.** 엔진의 OffloadingConnector 가 매 스케줄러 통계에 `kv_connector_stats` 를 실어 보내고, 프런트엔드 `KVConnectorLogging.observe`(`kv_connector/v1/metrics.py:71`)는 커넥터 클래스를 단언한다 → 첫 요청에서 `AttributeError: 'KVConnectorLogging' object has no attribute 'connector_cls'`, AsyncLLM 출력 핸들러 사망, HTTP 500(2차 원인). 클래스만 주면 Prometheus 쪽 `offloading/metrics.py:489` 가 `kv_connector_extra_config`(spec_name·티어)로 만든 지표 정의에 없는 키를 단언한다(3차 원인, AssertionError). → `kv_transfer_json.sh` 로 JSON 을 한 곳에서 만들고 프런트엔드는 `--kv-offloading-size` 없이 같은 JSON 을 받는다(97dc701849, 589e5afc0d). 노브가 비면 랭크 0/1 명령줄은 변경 전과 바이트 동일(드라이런 diff).

## 측정 (같은 순서: 부팅 → 유휴 샘플 → smoke → 8K×3 → 32K×3 → divergence → 유휴 3회)

| 항목 | 운영 baseline(19:30 부팅, 20:11) | 분리(20:37 부팅) | 운영 대조(20:52 부팅) |
| --- | --- | --- | --- |
| head 부모 프로세스 PSS/RSS | API 서버 396/560 MB(장기 가동), 부팅 직후 미측정 | headless 부모 1,011/1,277 MB | API 서버 1,015/1,282 MB |
| head MemAvailable 부팅 직후 | — | 5.67 GiB | 5.70 GiB |
| head MemAvailable 프로브 후 유휴(3회 평균) | 5.95 GiB | 3.96 GiB | 3.95 GiB |
| f323 MemAvailable 부팅 직후 / 프로브 후 | — / 8.02 GiB | 8.92 / 7.21 GiB (프런트엔드 PSS 965–987 MB) | 9.75 / 8.05 GiB |
| 8K TTFT 중앙값 (tok/s) | 4.50 s (1,822) | 4.55 s (1,799) | 4.54 s (1,802) |
| 32K TTFT 중앙값 (tok/s) | 19.49 s (1,682) | 19.46 s (1,685) | 19.52 s (1,679) |
| 8K 니들 정답 / greedy 반복 동일 | — | 정답 / True | 정답 / True |
| earlyoom 개입(20:00 이후) | 0 | 0 | 0 |

- 분리·대조 모두 32K 첫 실행이 느리다(24.2 s / 21.4 s; 부팅 후 첫 긴 콜드 프리필, dsv41.env 로더 주석의 알려진 현상). 중앙값 비교.
- 493K 세션은 head 유휴 3.96 GiB 로 4.5 GiB 게이트 미달이라 생략.
- greedy 동일성(divergence 4개 프롬프트): 분리 vs baseline 은 ko-busan·code 가 근접 로짓에서 argmax 가 갈림(수치형). 그러나 운영 대조 vs baseline 도 ko-food·ko-busan 이 같은 유형으로 갈리고(ko-food 42번 토큰은 로짓 차 3 nat 가 뒤집힘), 과거 운영 기록끼리(i25f vs i25f-2, i2f vs i24-base)도 ko-busan 이 갈린다. 즉 오늘 방법으로는 부팅·캐시 상태에 따른 운영 자체의 변동과 분리 구성의 영향을 가를 수 없다. code 만은 운영 기록 4개가 전부 117 토큰이고 분리에서만 99 토큰(53번 토큰 ' or'→':\n', 분리 쪽 로짓 차 0.125)이며, 분리 부팅의 첫 실행(smoke, 콜드)은 117 토큰으로 같았고 두 번째(프리픽스 히트)에서 갈렸다. 프런트엔드 쪽 요청 처리(토크나이즈·샘플링 파라미터)에 차이가 있다는 증거는 없다.

## 종료 상태(황금룰)

21:03 KST 채택 구성(노브 없음) TP=4, head :8889 health 200, 4대 `dsv41-serve` active, cap 4대 active·1989 MHz(status 1회 샘플 1995, 한도 2000 이내), head MemAvailable 5.31 GiB(70K 프리필 1회로 세그먼트 해제 후), earlyoom 0. 노드 체크아웃 4대 clean.

## 판단

- 프런트엔드를 head 밖으로 빼는 것만으로는 head 메모리를 돌려받지 못한다. 돌려받으려면 headless 부모 프로세스(`vllm serve --headless`, torch 를 임포트한 파이썬)를 없애거나 가볍게 해야 하며 이는 포크 변경이다(엔진 프로세스를 직접 부팅하는 얇은 런처, 또는 `run_headless` 가 torch 를 임포트하지 않도록).
- k8s 로 옮길 가치는 재핸드셰이크(프런트엔드 재시작 시 엔진 재접속) 없이는 없다. 둘 다 포크 변경 요구사항으로 후속 이슈에 적는다.
- 런처 노브는 기본 꺼짐으로 커밋(운영 기본값 불변). 채택 여부는 오너 결정.
