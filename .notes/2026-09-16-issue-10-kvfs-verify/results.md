# 이슈 #10 구현·검증 기록

## 구현 단계

- 기준: `5700ec2b8b`, 승인 계획 `plan.md`, 최신 레시피 `icmt-83bf7d7c`.
- 제품 변경: `deploy/gb10-cluster/dsv41/{kvfs_verify.py,test_kvfs_verify.py,README.md}`.
- 부가 변경: 이 메모와 계획, `.gitignore`의 메모 디렉터리 예외.
- 표준 라이브러리만 사용한다. 프로젝트에서 `re`를 금지하므로 경로는 문자열 분리·문자 검증으로 해석한다.
- 안전 조치: 기본 보고 모드, 서버 상태 불확실 시 삭제 거부, 링크를 따르지 않는 fd 접근, 삭제 직전 분류·파일 identity 재검사, 명시 후보 목록 출력. no_xattr/unknown/tmp는 삭제하지 않는다.
- 로컬 환경: uv로 생성한 Python 3.12 `.venv`, lint 의존성·pytest 설치, pre-commit/commit-msg 훅 설치.

### 실행 결과

```text
.venv/bin/python -m pytest -c /dev/null -p no:cacheprovider deploy/gb10-cluster/dsv41/test_kvfs_verify.py -q
23 passed, 4 skipped in 0.08s
```

macOS의 `os.setxattr` 부재로 실제 xattr 기반 분류/삭제 매개변수 케이스 4개는 skip했다. CRC 재읽기·지속 손상·fadvise는 속성 읽기 주입으로 로컬에서도 검사했다. 실제 Linux xattr 테스트 통과는 아직 주장하지 않는다.

```text
.venv/bin/pre-commit run --files deploy/gb10-cluster/dsv41/kvfs_verify.py deploy/gb10-cluster/dsv41/test_kvfs_verify.py deploy/gb10-cluster/dsv41/README.md .gitignore .notes/2026-09-16-issue-10-kvfs-verify/plan.md
All applicable hooks passed.

git diff --check
Passed.

.venv/bin/python deploy/gb10-cluster/dsv41/kvfs_verify.py --help
Exit 0; root/delete/limit/sample/progress/fadvise options displayed.
```

## validate 단계에 남은 사항

노드에는 아직 접근·반영하지 않았다. head pytest, 라이브/옛 저장소 전수 보고, 손상 사본 검출·삭제 거부, 실행 전후 메모리 및 운영 상태는 다음 단계에서 수행하고 여기에 추가한다. 실제 kvfs 파일 삭제는 현재 0이며 이후 검증에서도 0을 유지한다. 모델·엔진 변경은 없다.
