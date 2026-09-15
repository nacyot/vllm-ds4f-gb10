# #32 계획: dsv41-serve 영속 유닛, 워치독, 웜업, 구 DS4F 유닛 정리

정본은 #32 매니저 코멘트 `icmt-c89e6d68` 이다. 순서, 판정선, 제약, 파일 범위는 그 코멘트를 따른다. 이 문서는 파일을 읽고 노드를 읽기 전용으로 확인한 결과(2026-09-15 23:2x KST)로 구현 방식을 정한다.

## 1. 현재 동작과 문제점

- `dsv41_ctl.sh start`(`dsv41_ctl.sh:188-208`)가 노드마다 다음을 한다: `systemctl --user stop` → `shm_remote`(검증 목록만 지움) → `drop_caches` → `systemd-run --user --collect --unit=dsv41-serve $KNOBS`. 순서는 워커 27c4, 37cc, f323을 먼저 올리고 5 s 뒤 head를 올린다. 실험 knob은 `--setenv`로 넘긴다(`:15-16`). 공백이나 따옴표가 든 값은 이 경로를 통과하지 못한다(`dsv41.env:65` 주석).
- 4 노드 모두 transient 유닛이다. 확인값은 `FragmentPath=/run/user/1000/systemd/transient/dsv41-serve.service`, `Restart=no`이고 linger는 4 노드 모두 yes다. 재부팅하거나 프로세스가 죽으면 서버가 스스로 돌아오지 않는다. 15:56 earlyoom이 head 워커를 죽였을 때도 워커 3 대가 4 분 동안 남아 있었다.
- `/health`는 엔진의 errored 플래그만 본다. 워커 랭크가 죽고 head가 NCCL에서 멈춰 있어도 200을 돌려준다.
- 재부팅 위험이 있다. 37cc와 27c4의 `tp4-worker.service`는 enabled 상태이고 `WantedBy=default.target`이다. 이 유닛은 `ExecStartPre`에서 `rm -f /dev/shm/...*` 글롭 삭제를 한다. 재부팅하면 구 워커가 먼저 뜨고, `serve-node.sh:17-18`의 MemAvailable 100 GiB 게이트가 V4.1F 기동을 거부한다.
- 부팅 뒤 첫 8K 요청의 TTFT가 21 s다. JIT 커널과 flashinfer autotune 때문이다.
- 새로 확인한 사실은 다음과 같다.
    - systemd 255다.
    - head `/etc/fstab`에 `/mnt/kvdisk ext4 defaults,noatime,nofail`이 있다. head는 `/etc/exports`로 `/mnt/kvdisk`를 10.100.0.0/24에 내보낸다.
    - **워커 3 대는 head의 `/mnt/kvdisk`를 NFS4 `hard`로 마운트한다.** `serve-node.sh:135`의 `mkdir -p "$KVFS_DIR"`는 모든 랭크에서 실행되므로 head가 내려가 있으면 워커 기동이 NFS에서 멈출 수 있다.
    - 37cc는 `kv-prune.timer`가 enabled이고 active다. 27c4의 `ds4f-worker.service`는 이미 disabled다. f323에는 센티널만 있다.
    - head의 `ds4f-log-capture`(system), `nfs-server`, `gpu-clock-cap`, `earlyoom`은 모두 enabled다. earlyoom 인자는 `-m 2 -s 100 -r 300`이다.
- 구 DS4F 설계(head `~/.config/systemd/user/ds4f-head.service`와 드롭인, `~/migration-027/ds4f-pre.sh`, `ds4f-watchdog.sh`)를 읽었다. 가져올 것은 `Restart=always`, `KillMode=mixed`, 짧은 `TimeoutStopSec`, pre에서의 ssh 대기·마운트 대기·`reset-failed`, `TimeoutStartSec`를 올린 이유(pre 대기가 기본값 90 s에 죽음), 워치독의 두 가지 트리거 분류(결정적인 것 12회/2h, 헬스 기반 4회/2h에 쿨다운 15분), activating 장기 체류 알림이다. 가져오지 않을 것은 pre의 글롭 `rm`, swapoff/swapon(#30에서 cold anon swap-out이 오히려 여유를 만들었다), 로그 회전이다(요구는 같은 파일에 append).

## 2. 바꿀 것과 접근

### 2.1 유닛 이름과 랭크

유닛 이름은 `dsv41-serve.service`를 유지한다. 이 이름은 `shm_cleanup`, ctl, README, 골든룰 검사가 모두 참조한다. 4 노드가 **같은 유닛 파일**을 쓰고, 랭크는 호스트명으로 정한다. 지금 ctl에 있는 `RANK` 표를 공용 라이브러리로 옮겨 ctl과 유닛 헬퍼가 같은 표를 쓴다. 노드별 랭크 파일은 두지 않는다(드리프트 원천 제거).

### 2.2 새 파일과 바뀌는 파일

모두 `deploy/gb10-cluster/dsv41/` 아래다.

| 파일 | 내용 |
| --- | --- |
| `dsv41_lib.sh` (새로 만듦) | `dsv41_ctl.sh`에서 `RANK`/호스트 표, `cap_probe`, `check_caps`, `shm_cleanup`을 옮긴다. ctl과 유닛 헬퍼가 source한다. `cap_probe`는 대상이 자기 호스트이면 ssh 대신 같은 조회 명령을 로컬로 실행한다(head는 자기 자신에 ssh를 할 수 없다). `authorized_keys`는 건드리지 않는다. 조회 명령은 지금과 같다(`is-active`, `--query-gpu`만 쓰고 `-pm`/`-lgc`/`-rgc`는 없음). |
| `dsv41_unit.sh pre\|start\|post` (새로 만듦) | 유닛이 호출하는 헬퍼다. 출력은 모두 `~/dsv41-prep/logs/dsv41-r{r}.log`에 append한다. 파일은 지금과 같다. |
| `dsv41_watchdog.sh` (새로 만듦) | head 워치독(§2.5) |
| `dsv41_warmup.py` (새로 만듦) | 웜업 3 요청(§2.6). stdlib만 쓰고 torch는 쓰지 않으며 `kvoff_probe.build_prompt`를 재사용한다. |
| `systemd/dsv41-serve.service` (새로 만듦) | 4 노드 공통 |
| `systemd/dsv41-watchdog.{service,timer}`, `systemd/dsv41-warmup.service` (새로 만듦) | head 전용 |
| `dsv41_ctl.sh` | `start`/`stop`/`status`/`install` 변경(§2.4) |
| `selftest_caps.sh` | 새 흐름에 맞춰 수정하고 케이스를 추가한다(§4.1). |
| `README.md` | 유닛, 워치독, 웜업, 골든룰 검사 변경, 구 유닛 disabled 목록, 센티널 존재와 발동 수 |
| `serve-node.sh` | 변경하지 않을 예정이다. 게이트 대기는 pre가 맡는다. |

### 2.3 `dsv41-serve.service`와 `dsv41_unit.sh`

```ini
[Unit]
Description=DeepSeek-V4.1-Flash TP=4 rank (rank from hostname; head pre restarts the workers)
StartLimitIntervalSec=1800
StartLimitBurst=5
[Service]
Type=simple
EnvironmentFile=-%h/dsv41-prep/dsv41-override.env
ExecStartPre=/bin/bash %h/vllm-dsv41/deploy/gb10-cluster/dsv41/dsv41_unit.sh pre
ExecStart=/bin/bash %h/vllm-dsv41/deploy/gb10-cluster/dsv41/dsv41_unit.sh start
ExecStartPost=/bin/bash %h/vllm-dsv41/deploy/gb10-cluster/dsv41/dsv41_unit.sh post
Restart=always
RestartSec=20
TimeoutStartSec=1200
TimeoutStopSec=60
KillMode=mixed
SendSIGKILL=yes
LimitNOFILE=65536
[Install]
WantedBy=default.target
```

- `Restart=always`는 EngineDeadError 뒤 API의 종료 코드가 0일 수 있기 때문이다. `systemctl stop`은 재기동을 일으키지 않는다.
- `TimeoutStartSec=1200`은 head pre의 최악 경우를 덮는다. 마운트 대기 180 s, 캡 재시도 120 s, 워커 ssh 대기 합계 480 s, 워커 재기동 3 × 약 60 s를 더하면 약 960 s다.
- `pre` 공통 단계(모든 랭크):
  1. 이 노드에 유닛 밖 `vllm serve` 프로세스가 남아 있거나 MemAvailable이 100 GiB보다 작으면 최대 120 s 기다린다. 앞 인스턴스가 unified memory를 반환하는 시간이다. 그래도 조건을 못 채우면 exit 1.
  2. `shm_cleanup pre`
  3. `drop_caches`
- `shm_cleanup pre` 모드는 새로 만든다. 지금 `clean`은 유닛이 `activating`이면 건너뛴다. pre 단계에서는 유닛이 늘 activating이다. 그래서 `$INVOCATION_ID`가 `systemctl --user show dsv41-serve -p InvocationID`와 **같을 때만**(자기 유닛의 pre일 때) activating을 허용한다. 나머지 검증(검증 목록, 심볼릭 링크와 변경 파일 거부, `fuser`로 사용 중인 파일 제외)은 그대로 둔다.
- `pre` head(랭크 0) 추가 단계. 앞 단계가 실패하면 exit 1이고, 캡은 절대 만지지 않는다.
  1. `KVOFF_GIB≠0`이고 `KVFS_DIR`이 `/mnt/kvdisk` 아래이면 `mountpoint -q /mnt/kvdisk`를 최대 180 s 기다린다. 마운트가 늦으면 루트 NVMe에 `mkdir`하게 되는 경로를 막는다.
  2. `check_caps`로 4 노드를 확인한다(head는 로컬). 부팅 직후 cap 서비스가 activating일 수 있으므로 10 s 간격으로 최대 12회 재시도한다. 실패하면 `logger -p user.err`를 남기고 exit 1. `SKIP_CAP_CHECK`는 유닛으로 전달하지 않는다.
  3. head의 override 파일을 워커 3 대에 그대로 복사한다. **override의 정본은 head다.** 랭크마다 knob이 다르면 `DSV41_INDEXER_TP_SPLIT`처럼 부팅이 실패하는 경우가 있기 때문이다.
  4. 워커 27c4, 37cc, f323 순서로 ssh 도달을 기다리고(합계 480 s), `reset-failed`, `restart dsv41-serve`, `is-active`가 active가 될 때까지 대기한다.
  5. 5 s 기다린다.
- `start`: `exec serve-node.sh "$RANK"`
- `post`(head만): `systemctl --user start --no-block dsv41-warmup.service`. `Restart=`로 인한 자동 재기동을 포함해 매 기동마다 웜업이 확실히 걸리도록 `Wants=` 대신 이 방식을 쓴다.

### 2.4 `dsv41_ctl.sh`

- `start`:
  1. `check_caps`(워크스테이션, 지금과 같음)
  2. `KNOBS`를 `KEY="값"` 줄로 된 override 파일로 만들어 head `~/dsv41-prep/dsv41-override.env`에 쓴다. 전송은 base64로 한다. knob이 없으면 빈 파일이다. `DSV41_SHM_DRYRUN=1`도 여기에 적어 pre가 따르게 한다.
  3. FE가 설정돼 있으면 프런트엔드를 transient 그대로 기동한다(#24 스파이크, 기본 꺼짐).
  4. head에서 `systemctl --user restart dsv41-serve`를 실행한다. 워커 3 대는 head pre가 올린다.

  워커 선기동 순서는 유닛의 pre로 옮겨지므로, 순서 테스트는 pre 테스트로 옮긴다.
- `stop`: 지금 흐름(유닛 stop → pkill → 6 s → pkill -KILL → shm)을 그대로 둔다. inactive 상태의 유닛은 워치독이 건드리지 않는다.
- `install`(새로 추가): 4 노드에서 노드 체크아웃의 `systemd/dsv41-serve.service`를 `~/.config/systemd/user/`로 복사하고 `daemon-reload`, `enable`한다. head에는 워치독과 웜업 유닛을 더 복사하고 `enable --now dsv41-watchdog.timer`한다. 서버는 기동하지 않는다. 복사 방식은 `kvfs-gc` 선례를 따르고, 유닛 파일을 바꾸면 `install`을 다시 실행한다고 README에 적는다.
- `status`: 기존 행을 유지한다(selftest가 형식을 검사함). 끝에 `watchdog: <timer 상태>, attention: none|<사유>` 한 줄을 더한다.
- `shm`, `caps`, `headroom`, `log`, `frontend`는 동작을 바꾸지 않는다.

### 2.5 워치독(head, 1 분)

`dsv41-watchdog.timer`는 `OnBootSec=5min`, `OnUnitActiveSec=60s`로 두고, 서비스는 oneshot에 `TimeoutStartSec=300`이다. 상태는 `~/dsv41-prep/watchdog/`에, 로그는 `~/dsv41-prep/logs/dsv41-watchdog.log`에 둔다. 한 번 실행할 때 다음 순서로 판정한다.

1. `~/dsv41-prep/watchdog/paused`가 있으면 종료한다(벤치 중 프로브가 `/metrics`를 오염시키는 것을 막기 위함).
2. head 유닛 상태를 본다.
   - `inactive`: 운영자가 멈춘 것으로 보고 조용히 종료한다.
   - `failed`(시작 한도 소진): alert만 하고 재기동하지 않는다.
   - `activating`: 체류 시간을 기록하고, 1800 s를 넘으면 alert.
   - `active`: 다음 단계로 간다.
3. 유예: head `ActiveEnterTimestamp`에서 600 s가 지나지 않았으면 종료한다.
4. 워커 3 대를 ssh로 확인한다(각 20 s).
   - 유닛이 active가 아니거나, 워커 `ActiveEnterTimestamp`가 head보다 30 s 넘게 늦으면 결정적 트리거다.
   - 3회 연속 도달하지 못하면 헬스 트리거다.
5. `/health`가 3회 연속 200이 아니면 헬스 트리거다.
6. 5회에 한 번 1 토큰 추론 프로브를 보낸다(thinking false, temp 0, 120 s). 한 번 실패하면 다음 회차에도 프로브를 보내고, 2회 연속 실패하면 헬스 트리거다.
7. 조치는 `systemctl --user restart dsv41-serve` 하나다(head pre가 워커를 올린다). 결정적 트리거는 쿨다운 없이 12회/2h, 헬스 트리거는 쿨다운 15 분에 4회/2h다. 예산을 다 쓰면 alert.

alert는 `logger -t dsv41-watchdog -p user.err`와 `~/dsv41-prep/logs/dsv41-ATTENTION` 파일(시각과 사유를 덮어씀)에 남긴다. 이 파일은 사람이 확인하고 지운다.

### 2.6 웜업(head oneshot)

`dsv41-warmup.service`는 `After=dsv41-serve.service`, `TimeoutStartSec=1800`이고, 로그는 `~/dsv41-prep/logs/dsv41-warmup.log`다.

1. health 200을 최대 1500 s 기다린다. 그 사이 head 유닛의 InvocationID가 바뀌거나 active가 아니게 되면 "superseded"로 종료한다.
2. 3 요청을 차례로 보낸다. 요청마다 새 salt를 쓴다.
   - 약 8K 텍스트, greedy, 출력 8 토큰
   - 짧은 프롬프트, temp 0.6, thinking, 출력 32 토큰(rejection/resample 커널)
   - 코드로 만든 작은 PNG 1 장을 data URI로 넣은 요청
3. 요청마다 HTTP 코드, TTFT, 소요 시간을 한 줄씩 기록한다. 실패해도 유닛을 실패시키지 않는다. 기록만 하고 복구는 워치독이 판단한다.

### 2.7 구 유닛 정리

삭제하지 않고 disable과 stop만 한다. 목록은 README에 적는다.

- 조치 대상:
    - 37cc, 27c4의 `tp4-worker.service`: `systemctl --user disable --now`
    - 37cc의 `kv-prune.timer`: 같은 방식
    - head의 `ds4f-log-capture.service`: `sudo -n systemctl disable --now`
- 이미 disabled라 목록에만 적는 것:
    - head: `tp4-head`, `ds4f-head`, `ds4f-watchdog.timer`, `serve-warmup.service`(static), `kv-prune.timer`, `kv-snapshot-ttl.timer`
    - 27c4: `ds4f-worker`
- 그대로 두는 것: `uvm-stall-sentinel.timer`(4 노드), `kvfs-gc.timer`, `~/ds4f-logs`, `~/migration-027`, 구 유닛 파일

## 3. 영향 사이트

- 운영자와 에이전트의 `dsv41_ctl.sh start/stop/status` 사용. 명령 이름과 종료 코드는 그대로다. start는 이제 head 한 곳에서 restart하고, 순서는 유닛이 보장한다.
- 골든룰 검사 문구(README, 메모리 규칙)가 "`systemctl --user show dsv41-serve -p Environment` 비어 있음"에서 "override 파일 비어 있음 + `Environment` 비어 있음"으로 바뀐다.
- 실험 워커는 이제 knob을 override 파일로 넘긴다. 파일이 재부팅이나 자동 재기동 뒤에도 남으므로 **실험이 끝나면 기본값으로 `start`해야 override가 비워진다**. README 골든룰에 명시한다.
- 벤치와 `/metrics` 수집에는 워치독 프로브(5 분마다 1 토큰)가 섞인다. 벤치 중에는 `paused` 파일로 멈춘다.
- `dsv41-prep/logs/dsv41-r{r}.log`에 pre 로그 줄이 더해진다. `ctl log`는 그대로 쓸 수 있다.
- Grafana(CT116이 :8888 직접 수집): 웜업 3 요청과 프로브가 요청 수에 더해진다. 수가 작아 판정에는 영향이 없다.
- `serve-frontend.sh`와 #24 경로: 프런트엔드는 transient로 남는다. 순서는 프런트엔드 → head restart(pre가 워커를 올림)로 바뀐다.

## 4. 검증

### 4.1 로컬(워크스테이션, 노드 명령 없음)

- `selftest_caps.sh` 통과. 기존 케이스를 새 흐름에 맞춘다: start가 head에 restart 한 번만 보내는지, override 내용(빈 파일, knob 인용, DRYRUN)이 맞는지 본다. 추가할 케이스는 다음과 같다.
    - `dsv41_unit.sh pre` 랭크 0: 스텁 ssh, systemctl, nvidia-smi, mountpoint, sudo로 워커 순서 27c4→37cc→f323. 캡 실패, 마운트 없음, 워커 도달 불가일 때는 워커를 재기동하지 않고 exit≠0인지 본다.
    - `shm_cleanup pre`: InvocationID가 같으면 삭제하고 다르면 건너뛰는지 본다.
    - 워치독 판정표: inactive, failed, 유예, 헬스 3연속, 프로브 2연속, 워커 늦은 기동, 쿨다운, 예산 소진.
- `shellcheck`(pre-commit hook), `.venv/bin/python -m pytest deploy/gb10-cluster/dsv41/test_headroom.py`(회귀 확인)
- 노드에서 `systemd-analyze --user verify`로 유닛 파일 문법을 확인한다. torch를 쓰지 않는 명령이다.

### 4.2 클러스터(한 단계씩 포그라운드, 30~60 s 간격 관찰, 실패하면 거기서 멈추고 기본값으로 복구)

반영은 커밋 → `git format-patch` → 4 노드 `git am`으로 한다.

| 단계 | 조치 | 통과선 |
| --- | --- | --- |
| a 전환 | `dsv41_ctl.sh stop`(transient 제거) → §2.7 구 유닛 정리 → `install` → `start` | 4 노드 `FragmentPath=~/.config/systemd/user/dsv41-serve.service`, `Restart=always`, enabled. health 200. 웜업 로그 3 요청 200. 웜업 뒤 새 salt 8K 첫 요청 TTFT를 21 s와 비교해 기록한다(목표: 웜 상태 8K 수준 약 4.5~8.8 s). override 비어 있음. |
| b 프로세스 kill | head `VLLM::Worker_TP` 하나에 `kill -TERM` | 사람 개입 없이 4 랭크 유닛 ActiveEnter 갱신, health 200, 웜업 완료. 걸린 시간과 경로(Restart인지 워치독인지) 기록 |
| c 워커 재부팅 | 27c4 `sudo -n reboot` | 개입 없이 27c4 유닛 active, head 복구(결정적 트리거 또는 Restart), health 200과 프로브 200, 27c4 NFS 마운트 복귀 |
| d head 재부팅 | a~c가 모두 통과하고, fstab의 `/mnt/kvdisk`와 `nfs-server`/`gpu-clock-cap`/`earlyoom` enabled를 재확인한 뒤 1회 | 개입 없이 4 노드 serve active, caps active 1989(head는 직접 확인), health 200, 웜업 완료, 워커 3 대 NFS 마운트 복귀 |
| e 골든룰 | 기본값 상태 확인 | `dsv41.env` 기본값(override와 Environment 비어 있음), health 200, caps 4 대 active 1989, 노드 트리 clean, 워치독 timer active이고 attention 없음, head MemAvailable 기록. casebench solo S128 또는 bench2 C1 1회가 기준선 안(S128 1,792~1,836 tok/s, 디코드 스텝 68 ms). 벤치 중에는 워치독 paused |

결과는 `results.md`에 단계별 시각, 판정, 소요 시간, 이상 징후로 남긴다. 안전 확인창이 뜨면 전역 규칙대로 기록한다.

## 5. 이번에 하지 않을 것

- `uvm-stall-sentinel`의 동작 변경(README에는 존재와 7일 발동 수만 적는다. 판단은 #31), `~/dsv41-prep/kvfs` 삭제, `/mnt/kvdisk` 변경
- `gpu-clock-cap.service`에 대한 의존성 추가나 캡 변경. 캡 게이트가 실패하면 유닛 실패와 보고만 한다.
- 프런트엔드(#24) 유닛의 영속화, `vllm/` 코드 변경, 로그 회전, swap reset
- head `authorized_keys` 변경(로컬 조회로 대체)
- 화면 변경 없음

## 6. 오너 확인 (plan-approve)

1. **검증 창의 프로덕션 중단.** a~d에서 :8888이 네 차례 내려간다. 전환, kill 복구, 27c4 재부팅, head 재부팅이고, 예상 합계는 25~40 분이다. 매니저 코멘트가 "프로덕션 창"으로 정해 두었으므로 **승인 즉시 진행**을 기본으로 한다. 외부 사용자가 있는 시간대를 피해야 하면 시작 시각을 지정해 달라.
