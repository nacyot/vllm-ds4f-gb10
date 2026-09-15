# #40 — token_to_req_indices 커널 부분 이식 계획

## 목표와 근거

기존 token→request 매핑 생성만 업스트림 [PR 56562](https://github.com/vllm-project/vllm/pull/56562)의 Triton 구현으로 교체한다. 동일 빌드의 `DSV41_TOKEN_REQ_KERNEL` 게이트로 A/B하고, **코드 c1 디코드 스텝 3회 평균 개선율 ≥ 2.7%** 및 정확성 검증을 만족하면 배포 기본값으로 채택한다. 미달이면 구현을 main에 넣지 않고 결과만 보존한다.

계획 기준은 로컬 HEAD `8d9980dbf1`, Ryno #40 본문 및 매니저 코멘트 `icmt-a74257d5`다. 본문의 테스트 제외 지시는 후속 코멘트대로 해석하여 token mapping 테스트만 이식한다. PR의 GB200 전체 메타데이터 최적화 성능 수치를 GB10 기대 성능으로 사용하지 않는다. 매니저 프로파일의 스텝당 약 4.5 ms 호스트 비용은 프로파일러 부풀림을 포함한 가설 근거이며 채택 판정은 비프로파일 A/B로 한다.

## 현재 동작·문제점·불변 조건

`CommonAttentionMetadata.token_to_req_indices`는 device `query_start_loc`의 차분, `arange`, `repeat_interleave`, 버퍼 복사, 패딩 초기화를 수행한다. 여러 메타데이터 객체가 각자 생성하므로 객체 내부 캐시가 있어도 스텝 전체의 중복 launch는 남는다.

과거 결정 `7f7a32cfec`(DSpark adaptive verification)에서 CPU 경계 기반 구현을 device 기반으로 변경했다. CPU는 총 토큰 수가 맞지만 요청별 draft 경계는 오래된 값일 수 있다. 따라서 CPU 매핑으로 되돌리는 것은 허용하지 않는다.

- 실제 토큰의 요청 ID는 device 경계에서 결정한다. 길이 0 요청은 건너뛴다.
- `num_mapped_tokens = int(query_start_loc_cpu[-1])`라는 기존 총량 계약을 유지한다.
- 출력은 전달받은 int32 버퍼에 쓴다. `max(num_mapped_tokens, num_actual_tokens)` 용량 검사와 캐시 범위를 유지한다.
- 매핑 뒤 반환 범위 내 패딩은 0이며, 반환은 캐시의 `[:num_actual_tokens]` view다. 캐시 재사용 및 주소 안정성을 유지한다.
- CUDA graph replay에서 총량은 같고 device 요청 경계만 달라져도 새 경계가 반영되어야 한다.

## 변경 범위와 접근

| 파일 | 변경 |
| --- | --- |
| `vllm/v1/attention/backend.py` | `os` import와 모듈 상수 `_DSV41_TOKEN_REQ_KERNEL = os.environ.get("DSV41_TOKEN_REQ_KERNEL", "0") == "1"` 추가. 기존 캐시 처리 뒤 함수 내 분기로 on은 Triton, off는 기존 경로 실행. |
| `vllm/v1/attention/ops/metadata.py` | 기존 기능을 대체하는 업스트림 `_token_request` 이진 탐색 헬퍼와 `_token_request_mapping_kernel`만 이식. 256 토큰 블록, `num_warps=4`, `do_not_specialize` 유지. |
| `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` | 업스트림 `test_device_token_request_mapping`을 기존 suite에 추가하고 아래 계약 검증에 필요한 최소 매개변수·assert만 보강. |
| `deploy/gb10-cluster/dsv41/dsv41.env` | 설명 한 줄과 `DSV41_TOKEN_REQ_KERNEL=${DSV41_TOKEN_REQ_KERNEL:-0}` 추가. 채택 때만 기본 1을 별도 커밋. |
| `deploy/gb10-cluster/dsv41/dsv41_ctl.sh` | KNOBS의 `DSV41_INDEXER_TP_SPLIT` 옆에 게이트 이름 추가. |

`serve-node.sh`의 기존 `set -a`가 환경변수를 export하므로 수정하지 않는다. 게이트는 프로세스 시작 시 고정하며 A/B 사이 서버를 재시작한다. 일반 Python 기본값은 0, 채택 후 DSV4.1 배포 기본값만 1이다. 빈 출력은 불필요한 zero-grid launch 없이 같은 빈 view를 반환하도록 최소 처리한다.

max10 미커밋 초안은 참고만 하고 다른 이슈 변경을 복사하지 않는다. 업스트림 해당 코드와 출처를 기준으로 작은 diff를 만든다. Python/Triton 변경으로 네이티브 재빌드는 필요하지 않다.

## 영향 사이트

DSV4.1 직접 호출자는 `models/deepseek_v4_1/sparse_mla.py:192`, `models/deepseek_v4_1/compressor.py:109`, `v1/attention/backends/mla/sparse_swa.py:565`다. 결과는 sparse MLA 요청 인덱스, compressor ring slot mapping, SWA 메타데이터의 입력이고 DSpark draft/verification에서도 사용된다.

공통 함수의 추가 호출자는 `models/deepseek_v4/{sparse_mla,compressor}.py`, `models/qwen4_exp/common/qsa_cache.py`, `v1/attention/backends/mla/flashinfer_mla_sparse.py`다. 호출자 파일은 수정하지 않으며 전역 게이트 on이 공통 함수에 적용된다는 점을 검토한다. 기본 off는 기존 실행을 보존한다.

## 검증 방법

### 계약 및 정적 검사

테스트 설계: 모듈 목적은 device 경계의 request ID 매핑이며, 입력은 경계·토큰 수·재사용 출력 버퍼, 출력은 int32 view다. 방어할 실패는 0길이 요청의 잘못된 ID, 패딩 잔재, CPU 경계 오사용, replay 및 캐시 계약 손상이다. 기존 pytest의 공통 메서드 수준 테스트가 가장 작은 검증 단위다.

- 기존 매핑 테스트를 gate off/on 양쪽에 실행한다. 단일 요청, 중간·끝 0길이, 257 토큰 경계, 여러 요청, 전부 0길이 및 빈 출력, 패딩, mapped 수가 반환 수보다 큰 경우를 최소 매개변수로 확인한다.
- 정수 결과 원소 일치(`rtol=atol=0`), 출력 view 주소와 길이, 재호출 캐시 재사용, 쓰기 범위 밖 sentinel 보존을 확인한다.
- CUDA graph는 CPU 경계를 그대로 두고 device 길이 순서를 변경하여 replay 결과 및 패딩을 확인한다. 두 경로의 동작을 검증하도록 테스트에서 모듈 게이트를 통제한다.
- GPU 테스트는 **부팅 A 정지 후 부팅 B 전, 헤드 서버가 없는 구간**에서만 실행한다: `.venv/bin/python -m pytest tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py -k token_request_mapping -q`. 필요한 의존성은 `uv`와 기존 `.venv`로 관리한다.
- 변경 파일 대상 pre-commit, 셸 변경 `bash -n`, `git diff --check`. 이 단계에서는 실행하지 않으며 구현·검증 단계에서 결과를 기록한다.

### GB10 TP=4 실측·복구 (부팅 최대 3회)

검증 단계에서 원격 현재 상태와 기존 `~/sglang-cmp` 스크립트의 인자를 확인한다. 작업 전용 새 패치 디렉터리에서 `git format-patch`를 만들고 4대 `~/vllm-dsv41`에 `git am`한다. 노트/README는 배포 대상에서 제외하고 작업자 Git identity를 명시한다. 기존 노드 변경과 충돌하면 덮어쓰지 않고 해당 단계에서 해결한다.

1. **A / off:** 배포 기본 0으로 시작, health 확인. 기존 `run_suite.sh`와 동일한 solo 16K + mixed 6K+2 웜업. `decodebench.py --types prose,code --levels 1,2 --gen 256`을 `i40-off-r1..3`으로 3회 실행한다.
2. A를 정지하고 위 GPU 정확성 테스트를 실행한다. 실패하면 B 성능 측정으로 진행하지 않고 기본 프로덕션을 복구한다.
3. **B / on:** `DSV41_TOKEN_REQ_KERNEL=1 dsv41_ctl.sh start`, 동일 웜업과 `i40-on-r1..3` 측정. `longctx.py --records 10700` 1회로 약 158K 컨텍스트 답 `12`, 후속 답 `986`을 확인한다. 코드 수락 길이 5.82, 산문 기존 범위 2.91~3.12 및 A/B 출력·수락 길이 변화를 확인하고, 새 경고/오류를 `r0.log`에서 점검한다. 이상이 있으면 정확성 미통과로 판정한다.
4. 각 run의 `step_ms = tokens_per_chunk / per_stream_1 × 1000`을 계산한 뒤 코드 c1의 A/B 각 3회 산술평균으로 `100 × (off_mean - on_mean) / off_mean`을 판정한다. 과거 prod-i38 약 68.1 ms는 참고값이고 이번 A가 기준이다. 표에 raw r1~3, 평균·범위, 수락 길이, 처리량과 산문 c1·양쪽 c2도 함께 남긴다.
5. **C / 최종:** 정확성 통과 및 개선율 ≥ 2.7%면 배포 env 기본 1 별도 커밋을 반영하고 오버라이드 없이 시작한다. 미달·검증 실패면 후보 코드를 main에 넣지 않고 노드의 이번 후보 변경만 되돌려 기존 기본 구성으로 시작한다. 미달 결과는 `.notes/2026-09-15-issue-40-token-req-kernel/results.md`에 보존한다. 어느 경우든 :8888 health 200, `dsv41_ctl.sh caps` 4대 active/1989 MHz, `prod-i40` decodebench 1회로 최종 상태를 확인한다.

실험은 한 단계씩 포그라운드로 수행한다. 서버 가동 중 노드에서 pytest/torch를 실행하지 않는다. `nvidia-smi -pm/-lgc/-rgc`, clock-cap 서비스 정지, 임의 삭제를 하지 않는다. 성능 개선은 정확성 통과를 대신하지 않으며 경계값은 반올림 전 수치로 판단한다.

## 제외 범위와 승인

SM100 전용 `_indexer_decode_metadata_kernel`, `indexer.py` 및 fused indexer 테스트, #38 TP split, 호출자 간 캐시 공유/#41, 새 JIT 등록·튜닝·별도 마이크로벤치는 포함하지 않는다. 화면 변경은 없다.

오너만 정할 미해결 결정은 없다. 위 범위와 측정·채택/미채택 복구 절차를 plan-approve에서 일괄 검토받는다. 현재 단계 산출물은 이 계획서 하나이며 코드·커밋·서버는 변경하지 않는다. 승인 뒤 implement→validate→merge 워크플로를 따르고, main 반영은 merge 단계에서만 처리한다. 이슈 in_progress/done은 매니저 소유다.
