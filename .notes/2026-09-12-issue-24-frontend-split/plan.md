# 이슈 #24 계획 — API 서버(프런트엔드)를 head 밖으로 분리하는 타당성 스파이크 (2026-09-12)

워크트리 `issue-24-frontend-split`(main fb387ad984). 계획 시점 클러스터: 4대 `dsv41-serve` active, cap 4대 active·1989 MHz, head :8889 health 200, head MemAvailable 7.35 GiB(7,708,224 kB), f323 9.34 GiB, port 29560 은 head·f323 모두 비어 있음, head→10.100.0.32 경로는 `enp1s0f0np0`(200G). 노드 `~/vllm-dsv41` 은 4대 모두 clean, `d0a5826db3`(체크섬 테스트) 상당 커밋에 있고 main 의 마지막 커밋 fb387ad984 는 README 만 바꾼 것이라 노드에는 없다 → 이번 패치 시리즈는 `d0a5826db3..HEAD` 로 만들어 fb387ad984 를 함께 보낸다. 운영 API 서버(pid 3382394, EngineCore 의 부모) 유휴 RSS 560 MiB / PSS 396 MiB — 매니저 코멘트 icmt-166e0151 의 수치와 같다.

## 현재 동작과 문제

- `serve-node.sh`(`deploy/gb10-cluster/dsv41/serve-node.sh:171-175`)는 랭크 0 이면 `vllm serve … --host 0.0.0.0 --port $PORT`, 랭크 1~3 이면 `--headless`. 랭크 0 프로세스 하나가 API 서버(프런트엔드)이자 EngineCore 의 부모이고, 둘 다 head(gx10-6040) 에 산다.
- head 는 4대 중 가장 빡빡하다(유휴 7.3 GiB, 493K 프리필 바닥 4.82 GiB, earlyoom 선 2.43 GiB; #15·#25). API 서버가 나가면 head 는 PSS 기준 0.4 GiB, RSS 기준 0.56 GiB 를 돌려받고(부팅 초기의 1.2 GiB 는 정상 상태 값이 아님), 요청 상태·토크나이저·파서가 있는 프런트엔드는 KV 를 갖지 않으므로 원칙적으로 엔진과 다른 노드(나중엔 k8s)에 둘 수 있다.
- 질문은 "이 포크(v0.27 계열)에서 DP=1·TP=4 로 headless 엔진 + 원격 프런트엔드가 되는가"이며, 코드 변경 없이 런처 노브만으로 실기 확인하고 수치를 남기는 것이 이 이슈의 범위다.

## 코드 확인 (매니저 코멘트 근거를 워크트리에서 재확인, 전부 일치)

- `vllm/entrypoints/cli/serve.py:66-75` `--headless` → `api_server_count=0`; `:146-147` → `run_headless`. `:178-262` `run_headless` 는 `node_rank_within_dp > 0` 이면 다중 노드 TP 워커(랭크 1~3 의 현재 경로), 0 이면 `CoreEngineProcManager(local_client=False, handshake_address=tcp://<data_parallel_master_ip>:<data_parallel_rpc_port>)`. `data_parallel_size_local > 0` 필요 — nnodes=4 추론으로 1(`vllm/engine/arg_utils.py:2150-2165`).
- `arg_utils.py:2258-2274` `--data-parallel-address` 기본값은 `--master-addr`(=10.100.0.16, head 자신) → head 에서 **반드시** 프런트엔드 IP 를 명시.
- `vllm/v1/engine/utils.py:1173-1181` `dp_rank == 0` 이면 로컬 엔진 0개여도 `CoreEngine(0, local=False)` 와 핸드셰이크("zero local engines and rank 0 is headless" 주석), `:1195-1206` `handshake_local_only = (local_engine_count == dp_size)` = False → `tcp://<data_parallel_master_ip>:<rpc_port>` 에 ROUTER 바인드. 프런트엔드는 **자기 IP** 를 `--data-parallel-address` 로 준다. `:1350-1364` 원격 엔진은 `headless` 여야 함 → head 의 `--headless` 와 일치.
- `vllm/v1/engine/core.py:1314` `data_parallel = dp_size > 1 or dp_rank > 0` = False → DP=1 head 엔진은 `EngineCoreProc`(DP 그룹·코디네이터 없음, `:2073` 의 `dp_size > 1` 단언 경로 아님).
- `vllm/config/parallel.py:899` `data_parallel_size_local == 0` 을 DP 지정으로 취급. `:943-965` GPU 수 < world_size 검사는 `distributed_executor_backend is None` 일 때만 → 프런트엔드는 `--distributed-executor-backend mp` 명시. `:746-752` `nnodes_within_dp` 가 `data_parallel_size_local` 로 나누므로 프런트엔드에 `--nnodes` 를 주지 않는다(0 나누기).
- 프런트엔드가 갖지 않아도 되는 설정: `vllm/v1/spec_decode/metrics.py:211-213` `SpecDecodingProm` 은 `speculative_config None` 이면 비활성(관측 `:266-268` 조기 반환), `vllm/v1/metrics/loggers.py:115-117` KV 커넥터 로깅은 설정 없이도 생성. → 프런트엔드에는 `--speculative-config`/`--kv-*`/`--engram-config`/`--compilation-config` 를 주지 않는다. 대가: 프런트엔드의 `vllm:spec_decode_*` Prometheus 지표가 사라진다(엔진 로그의 acceptance 줄은 그대로). 스파이크에서는 이 손실을 기록만 한다.
- 토크나이저: V4.1 은 포크 토크나이저(`vllm/tokenizers/deepseek_v41.py`)가 모델 디렉터리의 `encoding/encoding.py` + `tokenizer.json`/`tokenizer_config.json` 을 읽는다(`chat_template` 키 없음, `generation_config.json` 없음 — head 도 동일). f323 은 head 와 같은 전체 모델 디렉터리를 갖는다. k8s 후속에는 `encoding/` 디렉터리도 필요하다고 적는다.
- Rust 프런트엔드는 `VLLM_USE_RUST_FRONTEND` 기본 False(`vllm/envs.py:164`) 라 이 경로에 관여하지 않는다.
- **핵심 리스크(매니저 지적 유지)**: 엔진은 기동 시 한 번의 핸드셰이크로 프런트엔드의 입력/출력 소켓 주소(`tcp://host:0` 커널 배정 포트, `utils.py:1045-1054`)를 받고 재핸드셰이크 경로가 없다. 프런트엔드 재시작 = 엔진 재기동일 가능성이 높다 → 실험 6단계에서 실기로 확정한다.

## 바꿀 것과 접근 (vLLM 파이썬 변경 없음, 런처 노브만)

노브가 비어 있으면 **오늘과 바이트 단위로 같은 명령줄·같은 동작**이 불변 조건이다. 운영 기본값(dsv41.env)은 분리 꺼짐.

1. `dsv41.env` — 노브 3개 추가(모두 기본 빈 값/미사용):
   - `FRONTEND_HOST=${FRONTEND_HOST:-}` 프런트엔드를 띄울 노드 호스트명(비면 분리 없음, 랭크 0 이 지금처럼 API 서버).
   - `FRONTEND_ADDR=${FRONTEND_ADDR:-}` 프런트엔드가 바인드하고 head 엔진이 접속할 IP(200G 링크, 예 10.100.0.32). `FRONTEND_HOST` 와 함께 설정.
   - `DP_RPC_PORT=${DP_RPC_PORT:-29560}` 핸드셰이크 ROUTER 포트(비어 있음 확인함).
2. `serve-node.sh` — 랭크 0 분기만 확장: `FRONTEND_ADDR` 가 비어 있지 않으면 `exec vllm serve "${ARGS[@]}" --headless --data-parallel-address "$FRONTEND_ADDR" --data-parallel-rpc-port "$DP_RPC_PORT"`, 비면 기존 줄 그대로. 랭크 1~3 분기·ARGS·환경변수는 손대지 않는다.
3. 새 `serve-frontend.sh` — venv·dsv41.env 는 serve-node.sh 와 같은 방식으로 source. 모델 디렉터리(`config.json`·`tokenizer.json`·`encoding/encoding.py`)만 확인하고, 100 GiB 메모리 게이트 대신 MemAvailable ≥ 4 GiB 게이트(프런트엔드 약 1 GiB, earlyoom 2.43 GiB 여유). `VLLM_ENGINE_READY_TIMEOUT_S=3600` 은 동일하게(엔진 부팅 약 150 s 를 기다림). 명령은 매니저 레시피 그대로:
   `vllm serve "$MODEL" --served-model-name "$SERVED_NAME" --tensor-parallel-size 4 --distributed-executor-backend mp --data-parallel-size 1 --data-parallel-size-local 0 --data-parallel-address "$FRONTEND_ADDR" --data-parallel-rpc-port "$DP_RPC_PORT" --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --block-size 64 --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 --default-chat-template-kwargs "{\"thinking\":$THINKING}" [--language-model-only] --host 0.0.0.0 --port "$PORT"` (+ `FRONTEND_EXTRA_ARGS`). `--nnodes/--node-rank/--master-*/--kv-*/--engram-config/--speculative-config/--compilation-config/--load-format` 은 주지 않는다. TP=4 가 프런트엔드에서 검증에 걸리면 `FRONTEND_TP=1` 폴백(프런트엔드는 엔진을 띄우지 않아 TP 값을 쓰지 않음).
4. `dsv41_ctl.sh`
   - KNOBS 목록에 `FRONTEND_HOST FRONTEND_ADDR DP_RPC_PORT FRONTEND_TP FRONTEND_EXTRA_ARGS` 추가(`--setenv` 로 head 와 프런트엔드 유닛에 전달).
   - `start`: 워커 3대 → (`FRONTEND_HOST` 설정 시) `FRONTEND_HOST` 에 유닛 `dsv41-frontend` 로 `serve-frontend.sh` 를 `systemd-run`(로그 `~/dsv41-prep/logs/dsv41-frontend.log`) → 5 s → head. 마지막 안내 줄은 프런트엔드 호스트:포트를 찍는다.
   - 새 하위 명령 `frontend`: 프런트엔드 유닛만 정지 후 같은 명령으로 재기동(6단계 재시작 실험을 재현 가능하게). `FRONTEND_HOST` 가 비어 있으면 거부.
   - `stop`: 4대 모두에서 `dsv41-frontend.service` 도 정지(노브와 무관하게 항상; 없는 유닛은 무시). 기존 pkill 패턴 `[v]llm serve.*DeepSeek-V4.1` 은 프런트엔드 명령줄(모델 경로 포함)도 잡으므로 그대로.
   - `status`: `FRONTEND_HOST` 설정 시 그 노드의 `dsv41-frontend` 상태 한 줄과 health 를 `FRONTEND_HOST:PORT` 로 curl. 비면 지금처럼 HEAD.
   - `log`: `log frontend [n]` 이면 `FRONTEND_HOST` 의 프런트엔드 로그.
5. `README.md` — Files 표에 `serve-frontend.sh`, 새 절 "Remote frontend (issue #24)": 레시피·순서·핸드셰이크 결합 제약·측정 결과 한 단락. Operating rules 에 "프런트엔드가 f323 에 있는 동안 f323 도 torch 프로세스 금지" 한 줄.

## 영향 사이트

- `serve-node.sh` 랭크 0 분기: 호출자는 `dsv41_ctl.sh start_rank`(systemd 유닛). 랭크 1~3 과 다른 스크립트는 ARGS 를 공유하지 않으므로 영향 없음.
- `dsv41_ctl.sh status/stop`: 운영자 수동 사용 + 황금룰 복구 절차. stop 이 프런트엔드 유닛을 항상 정지하도록 하는 것은 노브가 꺼진 상태에서도 부작용이 없다(유닛 부재).
- `prefill_probe.py --base`, `smoke.py <base>`, `divergence.py record <tag> --base`, `bench2.py` 는 URL 인자를 받으므로 변경 없이 프런트엔드 주소로 측정한다.
- 노드 4대 `~/vllm-dsv41`: 런처 스크립트 3개 + README + .notes 만 바뀌는 패치. `.so` 재빌드·venv 재설치 없음.

## 실험 순서 (한 단계씩, 30~60 s 모니터링; 매니저 5절 그대로)

1. 사전: `dsv41_ctl.sh caps`·status(4대 active·1989, health 200, head `/proc/meminfo` MemAvailable ≥ 4.5 GiB). 운영(head :8889)에서 baseline: API 서버 PSS(`smaps_rollup`), head·f323 MemAvailable 1분 간격 3회, `prefill_probe.py i24-base --tokens 8192 --runs 3`·`--tokens 32768 --runs 3`(첫 JIT 요청 제외), `divergence.py record i24-base`.
2. `dsv41_ctl.sh stop`. 워크트리 커밋을 4대에 `git format-patch d0a5826db3..HEAD | ssh … git am`(런처·README 만). 노드 `git status` clean·HEAD 일치 확인.
3. `FRONTEND_HOST=gx10-f323 FRONTEND_ADDR=10.100.0.32 dsv41_ctl.sh start` → 프런트엔드 로그에서 핸드셰이크 대기·HELLO/READY, head 로그(`dsv41-r0.log`)에서 "Launching 1 data parallel engine(s) in headless mode, with head node address tcp://10.100.0.32:29560" 와 핸드셰이크 성공 확인. 실패하면 오류 원문 기록 → `FRONTEND_TP=1` 로 1회 재시도.
4. `gx10-f323:8889/health` 200 → `smoke.py http://gx10-f323:8889` → 8K 요청 1개 정답 확인.
5. 측정 반복(1단계와 같은 방법, URL 만 프런트엔드): API 서버 PSS 는 head 에서 사라졌음을 확인(`pgrep`), f323 프런트엔드 PSS, head·f323 MemAvailable, 8K/32K TTFT, `divergence.py record i24-fe` + `compare i24-base i24-fe`. head 유휴 MemAvailable ≥ 4.5 GiB 이면 493K 세션 1회(`memlog.py` 로 head 바닥, earlyoom 0). earlyoom 개입 시 그 구성은 기각.
6. `dsv41_ctl.sh frontend`(프런트엔드만 정지→재기동) → head 엔진이 재접속하는지(예상: 안 됨, 새 프런트엔드는 HELLO 대기, 엔진은 옛 주소로 응답 시도). 결과·로그 발췌 기록. 재접속이 안 되면 전부 stop.
7. 전부 stop → 노브 없이 `dsv41_ctl.sh start`(채택 구성 TP=4, head :8889) → health 200, 캡 4대 active·1989, head MemAvailable ≥ 4.5 GiB, earlyoom 0(황금룰).

## 검증 방법

- 로컬(macOS, 노드 접속 없이): `bash -n` 3개 스크립트, `shellcheck`(있으면), 그리고 노브가 빈 상태에서 `serve-node.sh` 가 만드는 랭크 0 명령줄이 변경 전과 동일한지 `exec` 를 `printf '%q '` 로 바꿔 찍는 드라이런 비교(모델 파일·MemAvailable 게이트는 `MODEL`·게이트 우회 없이 통과 못 하므로, 스크립트를 함수화하지 않고 `bash -x` 출력의 인자 배열을 비교). `selftest_caps.sh` 는 dsv41_ctl.sh 의 caps 경로가 그대로임을 확인.
- 클러스터: 위 실험 3~7 단계. 완료 기준(매니저 6절): 원격 프런트엔드로 8K 정답 + greedy 출력 운영과 동일(divergence compare), head 유휴 MemAvailable 절감량·API 서버 PSS·8K/32K TTFT 차이·프런트엔드 단독 재시작 결과가 이슈 #24 코멘트에 기록, 종료 상태 7단계.
- 결과 기록: `.notes/2026-09-12-issue-24-frontend-split/results.md` + 이슈 #24 코멘트 + README 절.

## 결정 분기 (실기 결과에 따라)

- 되면: 위 런처 변경을 커밋(기본 꺼짐), README 절, k8s 후속 이슈 등록(요구사항: GPU 없는 포크 이미지 `VLLM_TARGET_DEVICE=empty` + 모델 `config.json`·`tokenizer*`·`encoding/` 파일, head 에서 도달 가능한 고정 IP(hostNetwork), 커널 배정 포트라 고정 포트 없음, 프런트엔드 재시작 = 엔진 재기동 결합(재핸드셰이크 코드 필요), Rust 프런트엔드는 원격 후보 아님).
- 안 되면: 정확한 오류와 이식 비용을 이슈에 적고 런처 변경은 커밋하지 않으며 #7 로 되돌린다.

## 이번에 하지 않을 것

- vLLM 파이썬 변경(재핸드셰이크·프런트엔드 사망 감지·고정 포트). 필요성이 드러나면 후속 이슈 요구사항으로만 적는다.
- k8s 매니페스트·이미지·클러스터 네트워크 설계. 워커 노드 프런트엔드로 타당성·수치만 확인한다.
- 프런트엔드에 `--speculative-config` 를 넘겨 spec 지표를 살리는 실험(CPU 노드에서 dspark 설정 생성이 되는지 미확인, 이 이슈의 질문이 아님).
- 운영 기본값을 분리 구성으로 바꾸는 것(오너 결정).

## 오너 확인 사항 (plan-approve 에서)

1. 프런트엔드를 **살아 있는 워커 노드 gx10-f323** 에 두는 것은 "서버가 살아 있는 노드에서 torch 임포트 프로세스 금지" 규칙의 예외다(매니저 코멘트가 이 배치를 지정, f323 유휴 9.3 GiB, 프런트엔드 약 1 GiB). 이 예외를 스파이크 기간에 한해 허용하는 것으로 진행한다 — 다른 판단이면 m2pro(포크 venv 필요, torch 임포트 가능 여부 미확인)로 바꾼다.
2. 스파이크가 되어도 **운영 기본값은 분리 꺼짐**으로 두고, 채택 여부는 절감량(기대 0.4~0.6 GiB)·TTFT 차이·재시작 결합 결과를 보고 오너가 정한다.
