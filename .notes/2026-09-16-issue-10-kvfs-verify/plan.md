# 이슈 #10 — fs 티어 CRC 운영 검증 절차 개선

## 기준과 현재 동작

- 기준 checkout: `5700ec2b8b`. 이슈 본문·전체 코멘트를 확인했으며 최신 매니저 코멘트 `icmt-83bf7d7c`를 우선한다. 과거 중단 작업은 재개하지 않고 이번 워크플로의 승인부터 따른다.
- 기존 손상 검사는 일회성 명령으로만 남아 있다. 이를 기존 fs 티어의 저장 형식과 운영 절차에 맞는 재실행 가능한 도구·테스트·README로 정착시킨다.
- `file_mapper.py`의 저장 규칙은 `<root>/<base>/config.json`과 `<base>_r<rank>/<hhh>/<hh>_g<idx>/<hash>.bin`이다. 현재 형식은 `format_version=2`, `blocks_per_file=1`이며 기대 크기는 config의 `group_bytes[idx]`다. 모델명이나 18개 그룹의 크기를 하드코딩하지 않는다.
- `tiering/fs/io.py`의 xattr은 `user.vllm_kv_crc32`, 값은 CRC32의 4바이트 big-endian이다. 임시 파일에 쓰고 rename하며 xattr과 데이터의 관측 시점이 달라질 수 있다. `05a8365138`의 결정 기록도 동시 재저장에 의한 오탐을 막기 위한 한 번 재읽기를 명시한다. xattr 부재는 런타임에서도 CRC 검사 생략 사유다.
- `kvfs_gc.sh`와 타이머는 30분마다 TTL/용량 정리와 빈 디렉터리 삭제를 수행한다. 검증 중 파일·디렉터리 소멸은 정상 경쟁으로 처리해야 한다.
- 매니저 관측상 현재 라이브 루트는 `/mnt/kvdisk/kv/dsv41`(약 105만 파일, 102 GiB), 옛 루트는 `~/dsv41-prep/kvfs`(37 GiB)다. 수치는 검증 단계에서 다시 기록한다. 서버는 :8888이며 검사 때문에 중지하지 않는다.

## 바꿀 것과 접근

### 1. `deploy/gb10-cluster/dsv41/kvfs_verify.py`

표준 라이브러리만 사용하는 독립 CLI로 작성한다. torch/vllm/C 확장을 import하지 않고 일반 buffered read를 사용한다. 기존 경로·CRC 규칙만 작게 옮기고 원본 위치를 명시한다.

- 인자: 위치 인자 root(명시값 → `KVFS_DIR` → `/mnt/kvdisk/kv/dsv41`), `--delete` 기본 꺼짐, `--limit N` 기본 무제한, `--sample K` 기본 1, `--progress`, `--fadvise/--no-fadvise` 기본 켬. 양수 옵션은 입력 검증한다.
- 최신 레시피가 이전 `--sample` 의미를 변경했으므로 **검사 간격**으로 사용한다. 디렉터리별 이름을 정렬한 깊이 우선 `scandir` 순회에서 K번째마다 검사한다(K=1 전수). 동일 트리에서는 열거 순서와 무관하게 같은 집합을 선택하며 트리 전체 목록은 메모리에 쌓지 않는다. 라이브 변경까지 동일 표본을 보장하지는 않는다. 분류별 예시 경로는 고정 최대 5개로 별도 처리한다. `--limit`는 실제 선택한 검사 파일 수에 적용한다.
- config를 읽어 base별 기대 크기를 구성한다. config 누락·잘못된 형식·지원하지 않는 포맷·그룹 범위 초과·해석 불가 경로는 `unknown`으로 보고한다. 심볼릭 링크는 따라가지 않고 검사/삭제 범위 밖 접근을 막는다.
- 분류 우선순위: 크기 오류 `size_mismatch`, xattr 없음 `no_xattr`, CRC 비교 후 `ok` 또는 재읽기로 확정한 `crc_mismatch`. 잘못된 길이의 CRC 속성은 `unknown`으로 보고하고 삭제하지 않는다. xattr 미지원은 부재와 함께 보고하되 원인을 표시한다. 권한·I/O 오류를 정상이나 손상으로 위장하지 않고 진단과 종료 코드 2로 불완전 검사를 알린다.
- `.tmp`는 읽지 않고 `skipped_tmp`. 탐색/stat/open/getxattr 중 소멸은 `vanished`, 불일치에서 제외한다. 디렉터리 소멸은 진단에서 구분하며 내부 파일 수를 추정하지 않는다. CRC가 다르면 새로 열어 데이터와 속성을 한 번 더 읽는다. 가능하면 동일 fd로 크기·속성·데이터를 관측한다.
- 읽은 fd를 닫기 전에 `posix_fadvise(..., POSIX_FADV_DONTNEED)`를 실행한다. 미지원 플랫폼은 경고 후 검사하며 `--no-fadvise`로 끌 수 있다. 처리량은 실제 읽은 바이트와 경과 시간으로 산출한다.
- 분류별 개수·관측 바이트·예시, 시간·MiB/s를 출력한다. `--progress` 지정 시 검사 50,000개마다 stderr에 진행을 표시한다. 표본 검사임을 보고서에 표시하고 `files`는 검사 시도한 bin 수, tmp와 소멸 디렉터리는 별도 집계한다.
- 마지막 줄: `kvfs_verify: root=… files=N ok=N size_mismatch=N no_xattr=N crc_mismatch=N unknown=N vanished=N skipped_tmp=N deleted=N secs=S mib_s=R`.
- 종료 코드는 size/crc 불일치 없음 0, 발견 1(삭제 후에도 발견 사실 유지), 잘못된 인자·없는 루트·검사를 끝낼 수 없는 오류 2, 삭제 거부 3. `unknown/no_xattr`는 0이어도 검사 보장 범위에서 제외됨을 명시한다.
- `--delete`는 주입 가능한 서버 검사 함수로 `systemctl --user is-active dsv41-serve.service`의 active 또는 `pgrep -f 'vllm serve'` 존재 시 거부한다. 확인 명령 실패로 중지 여부를 판단할 수 없어도 거부한다. 사용자에게 노출되는 검사 무시 옵션은 만들지 않는다. 보고 모드도 실행 중 서버에 대한 경고만 표시한다.
- 삭제 후보는 size/확정 CRC 오류만이다. 루트와 후보의 실제 경로·일반 파일 여부·범위·동일 파일 여부를 검증하고 `/`, 홈, 저장소 루트 및 링크 이탈을 거부한다. 절대 경로 후보 목록 전체를 먼저 출력하고 서버 상태 및 후보 상태를 삭제 직전에 다시 확인한 뒤 명시 목록의 파일만 하나씩 삭제·기록한다. 검사 후 대체되거나 바뀐 파일은 삭제하지 않는다. 글롭·디렉터리 삭제는 없다.

### 2. `test_kvfs_verify.py`

이 도구의 계약은 저장 트리 입력을 분류·요약하고 명시적으로 허용된 손상 파일만 삭제하는 것이다. 주요 실패는 정상 파일 오삭제, 경쟁으로 인한 오탐, 불완전 검사의 정상 보고다. 가장 싼 검증은 작은 임시 트리와 서버/파일 경쟁 주입을 사용하는 pytest다. 별도 도구이므로 지정된 새 테스트 파일을 만들되 `test_memlog.py`의 importlib 로딩과 torch 비의존 단언 방식을 재사용한다.

- config와 정상/1바이트 손상/크기 오류/xattr 없음 파일 및 tmp를 만들고 분류·집계·기본 보존을 확인한다.
- 서버 active 시 코드 3·삭제 0, 비활성 주입 시 손상 2개만 삭제. 실제 서버 검사 함수의 명령 실패 거부도 검증한다.
- 재읽기로 정상화되는 CRC 경쟁, 재확인 후에도 손상, 파일·디렉터리 소멸, unknown 경로/config, 범위 밖 링크와 검사 후 대체 파일 보호를 작은 케이스로 확인한다.
- CLI 기본 루트 우선순위·결정적 표본·limit·최종 요약 및 종료 코드를 확인한다. torch/vllm 미로딩을 단언한다. xattr 기능 없는 플랫폼/파일시스템에서는 해당 테스트만 사유를 밝히고 skip한다.

### 3. README 및 기록

Files 표의 gc 행 옆에 두 파일을 추가한다. 저장소 설명에서 검사 절로 연결하고 Operating rules에 실행 예시와 용도를 설명한다. 기본 창은 재기동 전 서버 정지 상태, 손상 의심 시 실행 중 서버에서는 보고 모드다. 삭제된 손상 블록은 MISS→재계산되며 no_xattr/unknown은 삭제하지 않는다. GC 소멸은 오류가 아님을 설명한다. torch-free의 지정 테스트는 이번 이슈 레시피에 따른 실행임을 명확히 한다. 화면 변경은 없다.

메모는 `.notes/2026-09-16-issue-10-kvfs-verify/{plan.md,results.md}`에 둔다. 구현 단계에서 레시피의 `.gitignore` 예외 한 줄도 반영한다. 제품 코드 변경은 지정된 deploy 세 파일뿐이다.

## 영향 사이트와 유지할 불변 조건

- 생산자: FileMapper, Python/C fs 저장 경로. 저장 규격·엔진 코드는 그대로 두며 이 도구가 해당 규격을 소비한다.
- 동시 소비자: fs 로더, GC, 운영자. 보고 모드는 파일 내용/xattr/config를 변경하지 않는다. 서버 요청이나 모델 출력의 동작을 변경하지 않는다.
- CLI 소비자: 운영자와 results.md의 집계. 고정 마지막 줄과 종료 코드로 실제 손상·검사 불가·삭제 거부를 구분한다.
- 라이브/옛 저장소에서 이번 작업이 삭제하는 파일은 **0개**다. 삭제 성공 테스트는 pytest 생성 파일만 사용한다. 실제 파일 사본에서는 손상 주입과 서버 활성 상태의 삭제 거부만 확인한다.

## 검증 단계 실행 계획

1. 로컬에 uv로 관리하는 Python 3.12 `.venv`를 준비해 독립 테스트·문법/import·변경 파일 lint를 확인한다. 시스템 Python/bare pip는 사용하지 않는다. macOS xattr skip을 통과로 포장하지 않고 head 실행으로 보완한다. 엔진 설치나 GPU 테스트는 이 독립 스크립트에 필요하지 않다.
2. 승인된 구현 커밋의 지정 deploy 파일만 새 임시 패치 디렉터리에 `git format-patch`로 만들고 head `~/vllm-dsv41`에 `git am`한다. 다른 히스토리를 reset하지 않는다. README 충돌은 노드 전송에서 제외할 수 있다. 기존 대상 확인 후 `~/dsv41-prep/kvfs_verify.py`를 정식 스크립트의 심링크로 연결한다.
3. head의 기존 venv Python으로 pytest를 한 번 실행한다: `~/vllm-dsv41-venv/bin/python -m pytest -c /dev/null -p no:cacheprovider --basetemp <새 ext4 임시 경로> deploy/gb10-cluster/dsv41/test_kvfs_verify.py`. basetemp는 `mktemp -d ~/dsv41-prep/i10-pytest.XXXXXX`로 이번 실행 전용 생성하고 pytest가 비우는 범위를 확인한다. 실행 전에 torch/vllm import 부재를 검사한다.
4. `dsv41_ctl.sh status`의 `Running 0 reqs` 확인 후 `KVFS_DIR=/mnt/kvdisk/kv/dsv41`로 기본 루트 전수 보고 검사 1회(`--progress`, limit 없음, sample=1). 실행 전후 MemAvailable/Cached와 전체 stdout/stderr, 종료 코드·시간·처리량을 기록한다. 포그라운드로 실행하고 30~60초 간격으로 관측한다. 손상이 있으면 개수·예시를 그대로 보고하며 삭제하거나 결과를 0으로 만들지 않는다.
5. 옛 저장소를 명시 root로 보고 검사 1회 실행하여 참고값을 기록한다. 두 검사는 순차 실행한다. 라이브 검사는 원자적 스냅샷이 아니며 시작 후 생긴 파일까지 완전 포괄한다고 주장하지 않는다.
6. `mktemp -d ~/dsv41-prep/i10-copy.XXXXXX` 아래 config와 실제 파일 3개를 상대 경로 유지·xattr 보존 복사한다. 한 파일을 1바이트 덮어써 `crc_mismatch=1`, 사본의 `--delete`가 코드 3·삭제 0임을 확인한다. 정리가 필요하면 실제 경로와 목록을 확인해 해당 사본만 명시적으로 삭제하고 결과를 기록한다.
7. 종료 시 :8888 health 200, status, 4대 캡 active·1989 MHz, 워치독 timer active/ATTENTION 없음, 오버라이드 파일 0바이트, head MemAvailable을 읽어 기록한다. 모델 출력에 영향을 주는 변경이 없어 모델 eval 대신 이 운영 상태 확인을 한다.

## 이번에 하지 않을 것 / 승인 사항

엔진·C 확장·GC 변경, 실제 kvfs 손상 주입/삭제, 옛 저장소 정리(#42), 서버 재기동, 워치독 정지, 캡·sysctl·swap·drop_caches 변경은 하지 않는다. 검증 중 이상은 기록하고 매니저에게 반환하며 재시작으로 복구하지 않는다.

오너만 결정해야 할 추가 질문은 없다. `--sample` 충돌은 최신 레시피 우선으로 위와 같이 해결했다. 이 계획만 제출하고 plan-approve에서 사람 승인을 기다린다. 구현·검증·머지는 이후 워크플로 단계가 소유하며 이슈 상태는 매니저가 소유한다.
