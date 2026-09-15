# #39 검증 — 통과

2026-09-16 04:48~04:50 KST, 메인이 직접 리뷰했다. 기준은 `d0f108eab6fc3fcecc73f5d1e3b2677045ced658`, 구현 커밋은 `9afcaf1ebc`다. 원래 요구사항, SSOT `icmt-c6218ef6`, 승인 계획과 전체 변경 파일 목록·결과·실행 증거를 대조했다. 운영 데이터 형식이나 영속 코드 변경이 없는 측정·기록 작업이므로 독립 리뷰어는 추가하지 않았다.

## 요구사항과 회귀 확인

- **범위:** 기준 ref 이후 엔진·테스트·커널·배포·기존 벤치 도구 변경 0. notes 디렉터리와 .gitignore 예외 한 줄만 변경했다. 노드 빌드는 시작/종료 동일하며 네 노드 엔진 트리도 같다. 로컬 main과 기존 배포의 장식자 위치 차이는 #46부터 있던 차이로 결과에 공개했다.
- **실행:** A/B/C/D override 및 네 랭크 split 로그 수를 재확인했다. 마지막 D는 knob 없이 시작했고 네 노드 override가 0 bytes다. 셀 명령·순서가 원문과 맞고 셀별 전경 실행 로그 32개가 모두 rc=0이다. 각 부팅의 세 자동 웜업 성공 증거도 있다.
- **결과:** 구성마다 casebench 7행, decode 6행으로 총 28/24행이다. 정확한 태그·종류·병렬도 조합, 성공 스트림 수, 생성 토큰 수와 양의 처리량을 집계 검사로 확인했다. 32회 health/ATTENTION 검사와 S128 16회의 시작 메모리 가드가 기록됐다.
- **분석:** S128 r2/r3 평균, C−A/D−B, B−A/D−C, 코드 c1 스텝 변환 및 r4−평균을 재계산한 결과가 summary.json·results.md와 일치한다. 순서차 부호는 +/+/+/−이므로 SSOT의 회차 편차 판정이 맞다. 과거 비교값·상호작용·GC 겹침의 제한도 포함했다.
- **증거와 소비자:** 원격 i39 결과 행과 i39.log의 SHA-256이 저장된 파일과 모두 일치한다([해시](validation-source-hashes.txt)). results.md의 파일 링크가 모두 존재한다. report_tables.py의 S128 빈칸은 태그 선택 규칙 때문이며 별도 회차 표로 보완한 것을 확인했다. 기존 도구의 입력 형식·과거 결과는 변경하지 않았다.
- **운영 복구:** 04:48:47~48 재조회에서 네 노드 serve active, NRestarts=0, override 0 bytes, Environment 공백, dirty=0, 캡 active/Enabled/1989 MHz였다. head health 200, watchdog active, ATTENTION 없음도 확인했다. 검증은 읽기 전용이며 재부팅·추가 부하·라이브 pytest/torch는 수행하지 않았다.

노드 재조회 증거: [6040](validation-gx10-6040.txt), [f323](validation-gx10-f323.txt), [37cc](validation-gx10-37cc.txt), [27c4](validation-gx10-27c4.txt).

## 검사

```bash
pre-commit run --from-ref d0f108eab6fc3fcecc73f5d1e3b2677045ced658 --to-ref HEAD
jq -s -f .notes/2026-09-16-issue-39-gate-ab/summarize.jq \
  .notes/2026-09-16-issue-39-gate-ab/casebench.jsonl \
  .notes/2026-09-16-issue-39-gate-ab/decode.jsonl \
  | diff - .notes/2026-09-16-issue-39-gate-ab/summary.json
/opt/homebrew/bin/bash -n .notes/2026-09-16-issue-39-gate-ab/run-cell.sh
```

모두 exit 0. 변경 ref 범위의 lint 원문은 [validation-lint.txt](validation-lint.txt)에 있다. 원본 로그의 CR·공백으로 생기는 전체 diff 공백 경고는 기록 보존에 따른 포맷 사항이며 통과를 막지 않는다.

## 범위 해석과 결론

원문은 삭제·drop_caches를 금지하면서 기존 start를 필수로 지정했다. 승인된 계획에서 명시한 기존 start 내부 cleanup/drop_caches 범위를 적용했고 실제 삭제된 네 공유메모리 파일도 숨기지 않고 결과·부팅 로그에 기록했다. 수동 삭제·캡 조작·노드 git am은 추가하지 않았다.

GC와 회차 변동 때문에 측정값을 보편적인 인과 효과로 확대하지 않는다. 이 제한은 결과에 이미 반영되어 있으며 요청한 네 구성 비교·순서 판정·복구 증거는 갖췄다. 실질 지적이나 필수 rework 없이 merge 단계로 진행한다.
