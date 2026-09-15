# #40 token request kernel 검증 결과

## 상태

**미채택: 코드 c1 평균 스텝 개선 0.304%로 2.7% 기준 미달.** 후보 코드·테스트·게이트를 revert했으며, 통합 대상은 계획과 이 결과 기록뿐이다. 최종 서버 복구 및 health 200, 4대 캡, prod-i40 요청 확인 완료.

## 변경 리뷰와 로컬 검사

- 기준: `8d9980dbf1d122a71040b85e5fa00f40ffeb7d7d`, 후보: `8fd917c8ae`.
- 최초 요청 및 승인 계획과 직접 대조: token request 헬퍼/커널만 이식, 기본 off 환경변수 게이트, 배포 KNOBS 전달, 기존 pytest 확장. SM100 indexer 변경과 호출자 변경 없음.
- device 경계, CPU 총량, 출력 view/캐시와 패딩 계약 보존. 빈 출력은 on 경로에서 launch 생략.
- `.venv/bin/pre-commit run --from-ref 8d9980dbf1d122a71040b85e5fa00f40ffeb7d7d --to-ref 8fd917c8ae`: 통과(해당 hook 실행, 나머지 skipped).
- `/opt/homebrew/bin/bash -n deploy/gb10-cluster/dsv41/dsv41_ctl.sh deploy/gb10-cluster/dsv41/dsv41.env`: 통과.
- `.venv/bin/python -m py_compile vllm/v1/attention/backend.py vllm/v1/attention/ops/metadata.py tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`: 통과.
- `git diff 8d9980dbf1..HEAD --check`: 통과.

## 노드 반영

4대 변경 전 작업 트리 clean. 이전 배포는 #38 기본 채택 상태. 후보 patch는 `.notes/**` 및 README 제외 후 `git am`으로 적용했다. 로컬 패치: `/tmp/i40-token-req.grHXbI/0001-token-req.patch`, 노드: `/tmp/i40-token-req-8fd917c8ae.patch`.

| 노드 | 이전 HEAD | 후보 HEAD |
| --- | --- | --- |
| gx10-6040 | 9116567825 | ffe8aa283e |
| gx10-f323 | b27584dc9 | 13ee708ec |
| gx10-37cc | 87ca12ba9 | ed76f85db |
| gx10-27c4 | 72c0742ec | 6fb3f1442 |

## 운영 기록

- 2026-09-15 17:47~17:48 KST: 기존 서버 정지. cap service 4대 active, 부하 시 1989 MHz 확인. 종료 중 resource_tracker 경고는 기존 off 부팅에서도 존재.
- 최초 stop은 `DSV41_SHM_DRYRUN=1`로 대상 확인만 했다. 헤드에 `/dev/shm/vllm_offload_362833fb-5990-4459-bce7-38ea98561e46.mmap`(2,147,422,208 bytes) 하나가 미사용으로 남음. 나머지 3대 대상 없음.
- 17:48:58 KST A 시작: 기존 ctl의 shm_cleanup이 위 경로의 realpath, 소유자, 서비스 inactive, fuser 미사용, inode 동일성을 재확인한 뒤 해당 파일 하나 삭제. 메모리 조건 복원을 위해 승인된 start 정리 절차를 사용했으며 안전 확인창은 발생하지 않음. 새 서비스 부팅 재개 확인. 클록 설정/서비스는 변경하지 않았다.

## 실측

A: `DSV41_TOKEN_REQ_KERNEL` override 없음, 배포 기본 0. 17:48:58 KST 시작. 결과는 `~/sglang-cmp/results/`의 `i40-*` 태그로 기록한다.

### CUDA 정확성

A 종료 후 4대 inactive를 확인하고 헤드에서 `cd ~/vllm-dsv41 && .venv/bin/python -m pytest tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py -k token_request_mapping -q` 실행: **36 passed, 4 deselected, 14 warnings, 9.51 s**. 경고는 기존 torch.jit.script_method deprecation이다. off/on × 경계 6종 × 반환 패딩 3종; 정수 원소 일치, 버퍼 범위·캐시, stale CPU/device 변경 CUDA graph replay 통과.

A 코드 c1 step ms: 69.409660 / 67.533070 / 67.345522, 평균 **68.096084 ms**. 수락 길이 모두 5.82. API 클라이언트 스크립트는 표준 라이브러리만 사용하며 기존 노드 `.venv/bin/python`으로 실행했다. 성능 기록 뒤 코드/산문 각 2회 고정 프롬프트 응답을 별도 수집했다(`/tmp/i40-off-output.jsonl`); 이 시간은 성능 표에 포함하지 않는다.

A 종료 후 미사용 offload `/dev/shm/vllm_offload_0640cdeb-9f6f-4b69-94fa-611179a24514.mmap` 2,147,422,208 bytes를 확인했다. B 시작 때 기존 ctl 안전 검사 후 해당 파일만 정리했고 새 부팅 재개를 확인했다. 안전 확인창은 발생하지 않았다. B 로그는 `dsv41-r0.log` 88889번째 줄부터다.

### A/B raw 요약

각 run은 prose/code × c1/c2 × gen 256. 모든 스트림 성공, errors=[] 및 completion_tokens=256. step_ms는 기록된 tokens_per_chunk / per_stream_1 × 1000이며 아래 표시는 소수 셋째 자리로 반올림했다. 판정에는 표 반올림 전 값을 사용했다. 원본 벤치가 입력 필드를 소수 둘째 자리로 저장하는 한계는 있다.

| tag | 종류 | c | tok/s/stream | tokens/chunk | step ms |
| --- | --- | --- | --- | --- | --- |
| i40-off-r1 | prose | 1 | 33.32 | 3.01 | 90.336 |
| i40-off-r1 | prose | 2 | 30.83 | 3.08 | 99.903 |
| i40-off-r1 | code | 1 | 83.85 | 5.82 | 69.410 |
| i40-off-r1 | code | 2 | 62.29 | 5.82 | 93.434 |
| i40-off-r2 | prose | 1 | 37.66 | 3.08 | 81.784 |
| i40-off-r2 | prose | 2 | 30.43 | 2.98 | 97.930 |
| i40-off-r2 | code | 1 | 86.18 | 5.82 | 67.533 |
| i40-off-r2 | code | 2 | 62.49 | 5.82 | 93.135 |
| i40-off-r3 | prose | 1 | 40.70 | 3.01 | 73.956 |
| i40-off-r3 | prose | 2 | 30.16 | 2.98 | 98.806 |
| i40-off-r3 | code | 1 | 86.42 | 5.82 | 67.346 |
| i40-off-r3 | code | 2 | 62.07 | 5.82 | 93.765 |
| i40-on-r1 | prose | 1 | 38.91 | 3.20 | 82.241 |
| i40-on-r1 | prose | 2 | 29.64 | 2.93 | 98.853 |
| i40-on-r1 | code | 1 | 83.22 | 5.82 | 69.935 |
| i40-on-r1 | code | 2 | 61.61 | 5.82 | 94.465 |
| i40-on-r2 | prose | 1 | 40.09 | 3.01 | 75.081 |
| i40-on-r2 | prose | 2 | 27.09 | 2.81 | 103.728 |
| i40-on-r2 | code | 1 | 86.88 | 5.82 | 66.989 |
| i40-on-r2 | code | 2 | 77.09 | 5.82 | 75.496 |
| i40-on-r3 | prose | 1 | 42.23 | 3.05 | 72.224 |
| i40-on-r3 | prose | 2 | 30.12 | 2.94 | 97.610 |
| i40-on-r3 | code | 1 | 87.20 | 5.82 | 66.743 |
| i40-on-r3 | code | 2 | 77.48 | 5.82 | 75.116 |

| 종류 | c | off 평균 ms | on 평균 ms | 개선율 |
| --- | --- | --- | --- | --- |
| prose | 1 | 82.025432 | 76.515225 | 6.7177% |
| prose | 2 | 98.879578 | 100.063592 | -1.1974% |
| code | 1 | 68.096084 | 67.889060 | 0.3040% |
| code | 2 | 93.444648 | 81.692506 | 12.5766% |

**미채택:** 코드 c1 개선율 **0.30401699% < 2.7%**. off 범위 67.346~69.410 ms, on 66.743~69.935 ms로 겹친다. 코드 c2의 이득은 채택 기준이 아니며 on r1/r2 사이 편차도 크다. 산문 c1 on r1 수락 길이 3.20은 기존 참고 범위 2.91~3.12 밖이라 수락 길이 불변을 보편적으로 주장하지 않는다. 이 결과로 범위를 넓혀 튜닝하거나 추가 부팅 A/B를 하지 않는다.

B는 17:57:14 KST 시작, 17:59:41 health 200 확인. solo/mixed 웜업 후 메모리 여유 4.85 GiB(A mixed 후 4.72 GiB). 서비스 및 실제 실행 환경에서 게이트를 확인했다(A=0, B=1). 가중치·나머지 배포 노브·클록 캡 설정은 동일하다.

### 모델 응답 및 로그

- `longctx.py --base http://127.0.0.1:8888 --tag i40-on --records 10700 --out results/longctx.jsonl`: cold 158,515 토큰, TTFT 76.154 s, 2,081.5 tok/s, 답 `12` 정답. 후속 158,533 토큰, TTFT 0.647 s, 답 `986` 정답.
- decodebench 고정 프롬프트를 temperature=0/top_p=1/thinking=False/ignore_eos=True/max_tokens=256으로 비스트리밍 2회씩 별도 실행. 코드 off 2회/on 2회 텍스트가 모두 동일(726자). 산문은 off 자체 반복도 불일치(1,079/911자), on 자체 반복도 불일치(1,112/983자). 모든 호출 256 토큰 완료. 산문 차이를 이 커널의 인과 효과로 단정하지 않으며 전체 출력 동일성도 주장하지 않는다. B 응답 스냅샷은 긴 컨텍스트 검사 뒤라 캐시 이력도 A와 다르다.
- B 구간 ERROR/Traceback 없음. 새로운 token mapping JIT 경고 없음. 기존 A에도 있는 FlashInfer mxfp8 tuning-bucket fallback 계열 경고가 입력 shape 차이 및 B-only longctx에서 나타났다. 경고 문자열 전체가 동일하다고 주장하지 않는다.
- 지속 보존한 헤드 자료: `~/dsv41-prep/i40-8fd917c8ae/{i40-off-r0.log,i40-on-r0.log,i40-off-output.jsonl,i40-on-output.jsonl}`. 측정 원본은 `~/sglang-cmp/results/{decode,casebench,longctx}.jsonl`의 `i40-*` 태그.

## 최종 복구

성능 미달에 따라 구현·테스트·배포 게이트 변경 5개 파일만 되돌리고 계획·결과를 남긴다. 원래 기준 대비 `vllm/`, `tests/`, `deploy/` diff가 없는 것을 확인했다. 최종 운영 검사는 아래에 기록한다.

- 노드 revert 커밋: gx10-6040 `a700f91262`, gx10-f323 `ddbb488a2`, gx10-37cc `c9c752476`, gx10-27c4 `99b092a98`. 각 노드에서 `git diff HEAD~2 HEAD --exit-code` 통과(실험 전 파일 내용과 동일).
- 로컬 revert `c9beb11f22`. 변경 파일 pre-commit 및 sign-off hook 통과. 코드/테스트/배포 diff는 원래 기준 대비 0이다.
- B 종료 뒤 헤드 미사용 `/dev/shm/vllm_offload_ab7224a5-fb19-4358-9c4b-d253519996be.mmap` 2,147,422,208 bytes 확인. C 시작의 기존 ctl이 경로·소유자·미사용·inode 동일성 재확인 후 해당 파일 하나 삭제했다. 안전 확인창 없음, 실제 부팅 재개 확인.
- C 시작: 18:07:41 KST. `systemctl --user show dsv41-serve -p Environment`는 빈 값, 즉 실험 오버라이드 없음. 총 부팅 A/B/C 3회.

- 18:10:31 KST C health 200. 최종 요청 후에도 health 200, Environment 비어 있음, 헤드 작업 트리 clean.
- `dsv41_ctl.sh caps`: 4대 모두 service active / persistence Enabled / 1989 MHz(최종 요청 실행 중).
- `prod-i40` decodebench prose,code × c1,c2 × gen 256: 모든 스트림 성공, errors=[], 토큰 수 256. 코드 c1 85.07 tok/s, 5.82 tokens/chunk; c2 62.22 tok/s, 5.82. 산문 c1 49.98 tok/s, 3.37 tokens/chunk; c2 33.57 tok/s, 3.05. 최종 확인은 decodebench 내 32토큰 웜업만 사용했으므로 A/B 수치에 합치지 않는다. 원래 코드에서도 산문 수락 길이가 참고 범위를 벗어나며, 이 변동의 원인은 이번 범위에서 귀속하지 않는다.

## 인계

**검증된 통합 대상은 계획/결과 문서만**이다. 구현 커밋 `8fd917c8ae`는 로컬 `c9beb11f22` 및 4대 revert로 취소했다. 기준 ref 대비 코드·테스트·배포 변경이 없으므로 미채택 조건(죽은 게이트를 남기지 않음)을 충족한다. main 머지와 이슈 상태 변경은 이 검증 단계에서 실행하지 않았다. AI assistance: OpenAI Codex가 구현, 검사, A/B 실행 및 기록을 수행했다.
