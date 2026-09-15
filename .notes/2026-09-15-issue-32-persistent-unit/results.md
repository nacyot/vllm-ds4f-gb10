# #32 결과: dsv41-serve 영속 유닛, 워치독, 웜업

실측은 2026-09-15 23:59 ~ 2026-09-16 KST에 했다. 계획은 같은 디렉터리의 `plan.md`이고, 순서와 판정선의 정본은 #32 코멘트 `icmt-c89e6d68`이다. 노드에는 코드 커밋 2개만 반영했다(`deploy(dsv41): persistent dsv41-serve units ...`, `fix(dsv41): count the reasoning delta ...`). 방법은 `git format-patch`와 `git am`이다. 노드 README는 main보다 앞선 판이라 문서 커밋은 저장소에만 있다.

## 1. 판정

**통과했다. 영속 유닛, 워치독, 웜업을 4 노드에 반영했고 프로덕션을 복구했다.** 이슈의 완료 기준을 모두 채웠다.

- head 재부팅(d)과 워커 27c4 재부팅(c 2차) 뒤 사람 개입 없이 4 노드 serve active, caps active 1989 MHz, health 200, 웜업 완료까지 돌아왔다.
- head 워커 프로세스를 kill(b)한 뒤 자동으로 복구됐다(`Restart=always`).
- 구 유닛의 disabled 목록을 README에 적었다.

검증 중 실질 결함 2건을 찾아 그 자리에서 고쳤다.

1. **c 1차 실패.** 워커를 재부팅하면 `/mnt/kvdisk` NFS 마운트가 돌아오지 않는다. 그래서 rank 3의 `mkdir -p $KVFS_DIR`가 실패했고, start limit에 걸려 서버가 약 16 분 동안 내려갔다. 마운트를 손으로 되살린 것이 사람 개입이었다. `serve-node.sh`가 fs 티어 디렉터리를 rank 0에서만 만들게 고친 뒤, c 2차는 개입 없이 5 분 만에 복구됐다.
2. **웜업 기록 오류.** sampled-thinking 요청의 첫 토큰이 `delta.reasoning`에 실려 오는데 이를 세지 못해 `ok=false`로 기록했다. 고쳤다.

측정한 수치는 다음과 같다.

| 항목 | 결과 |
| --- | --- |
| 복구 시간 | kill 241 s(health 226 s), 워커 재부팅 302 s(워치독 결정적 재기동), head 재부팅 약 4 분 |
| 웜업 효과 | 부팅 뒤 첫 클라이언트 8K 요청 TTFT가 21.0 s(09-13)에서 웜업 뒤 5.34 s로 줄었다. 웜업 자신의 8K 요청은 부팅 직후 캐시 상태에 따라 6.0 ~ 19.9 s였다. |
| 골든룰 | 기본값, health 200, caps 1989 MHz, 노드 트리 clean, S128 1,894 tok/s, 디코드 스텝 68.2 ms |

계획과 다르게 한 것:

- 워치독 일시정지는 `paused` 파일 대신 타이머 stop/start로 한다. 파일을 지우려면 `rm`이 필요해 운영 규칙과 부딪치기 때문이다.
- 웜업은 `kvoff_probe`를 import하지 않고 같은 모양의 프롬프트를 스스로 만든다. `kvoff_probe`는 모듈을 불러오는 순간 서버에 요청을 보낸다.
- pre의 이전 인스턴스 대기는 MemAvailable만 본다. #24 프런트엔드가 f323에서 `vllm serve`로 뜨는 경우와 충돌하지 않게 하기 위해서다.

남는 것(오너 판단):

- 워커 3 대의 `/mnt/kvdisk` NFS 마운트는 부팅 때 복원되지 않는다. V4.1F에는 필요 없고 27c4는 지금 마운트가 없다. fstab에 넣을지는 오너가 정한다.
- 워치독 `restart-history`에 이번 시험의 결정적 재기동 3 건이 남아 있다(예산 12, 2 시간 창). 02:44 KST 이후 창에서 빠진다.

## 2. 로컬 검증

- `selftest_caps.sh`: 종료 코드 0, PASS 50. start 흐름, override 인용, install, head pre의 워커 순서와 실패 경로, shm `pre` 모드, 워치독 규칙을 다룬다.
- `pre-commit run --from-ref 803a4f88b7 --to-ref HEAD`: 모두 통과했다(shellcheck, ruff, 금지 import 포함).
- `test_headroom.py`: 13 passed.
- head `systemd-analyze --user verify`: 유닛 파일 4개 모두 통과했다.
- head에서 `systemd-run --user --pipe -p EnvironmentFile=`로 override 인용을 확인했다. `EXTRA_ARGS="--x \"a b\" \$y \\z"` 줄이 `--x "a b" $y \z`로 들어온다.

## 3. 클러스터 단계

### a. 전환 (transient → 영속)

| 시각 | 사건 |
| --- | --- |
| 23:59:28 ~ 00:00:01 | `dsv41_ctl.sh stop`. head의 `vllm_offload_*.mmap` 2 GiB 1개를 명시 삭제했고, 다른 노드에는 후보가 없었다. |
| 00:00 | 구 유닛 disable(§4 표). 4 노드의 transient `dsv41-serve`가 사라진 것(`LoadState=not-found`)을 확인했다. |
| 00:00:26 | `dsv41_ctl.sh install`: 4 노드 enable, head `dsv41-watchdog.timer` active |
| 00:00:39 | `dsv41_ctl.sh start`: caps 4 대 active, 정지 상태 SM 208~305 MHz, override 0줄 |
| 00:00:46 | head pre 시작 |
| 00:00:52 / 54 / 55 | 워커 27c4, 37cc, f323 재기동. 워커 pre는 각 1 s였다. |
| 00:01:06 | rank 0 exec. `start`가 반환됐다. |
| 00:03:23 | health 200. head active로부터 2 분 17 초, `start`로부터 2 분 44 초이고, head MemAvailable은 5.26 GiB였다. |
| 00:03:21 ~ 35 | 웜업 3 요청(표 아래) |
| 00:04:04 | 웜업 뒤 새 salt 8K(11,212 토큰) 첫 요청의 TTFT는 **5.34 s**다. 2026-09-13 부팅 뒤 첫 요청은 21.0 s였다. |

웜업 결과(부팅 뒤 첫 요청들):

| 요청 | TTFT | 결과 |
| --- | ---: | --- |
| text-8k-greedy (11,212 토큰) | 10.27 s | 답 `12` 정답 |
| sampled-thinking (32 토큰 생성) | 기록 안 됨 | 요청은 성공했지만 스트림의 사고 텍스트가 `delta.reasoning`에 실려 첫 토큰으로 세지 못했고 `ok=false`로 기록됐다. `fix(dsv41): count the reasoning delta ...`로 고쳤다. |
| image (PNG 1 장, 줄무늬 4개) | 0.45 s | 답 `4` 정답 |

기동 뒤 4 노드 상태는 다음과 같다.

- `FragmentPath=~/.config/systemd/user/dsv41-serve.service`, `Restart=always`, enabled, `NRestarts=0`
- `Environment` 비어 있음, override 0 bytes
- 노드 트리 clean

### b. head `VLLM::Worker_TP0` kill -TERM (earlyoom 재현)

| 시각 | 경과 | 상태 |
| --- | ---: | --- |
| 00:05:11 | 0 | `kill -TERM 1846157` (VLLM::Worker_TP0) |
| 00:05:26 | +15 s | `activating/auto-restart` |
| 00:05:39 | +28 s | head pre 시작(`NRestarts=1`) |
| 00:05:45 / 59 / 06:13 | | 워커 27c4, 37cc, f323 재기동. 각 약 14 s이며, 죽은 피어 옆 워커의 stop이 걸린 시간이다. |
| 00:06:31 | +80 s | rank 0 exec |
| 00:08:57 | +226 s | health 200 |
| 00:08:59 | +241 s | 웜업 3 요청 모두 ok. 8K TTFT 5.95 s, sampled-thinking 0.34 s, image 0.45 s |

- 사람 개입은 없었다. 복구 경로는 `Restart=always`였다. 워치독은 유예 중이라 조치하지 않았고 `restart-history`도 없다.
- 복구 뒤 워커 3 대의 ActiveEnter(…758, …772, …785)가 head(…791)보다 앞선다. head MemAvailable은 4.81 GiB였다.

### c. 워커 27c4 재부팅

#### 1차: 실패(사람 개입이 필요했다), 수정 뒤 다시 시험

| 시각 | 사건 |
| --- | --- |
| 00:17:28 | 가드를 확인했다(head active 657 s, health 200, 4 노드 active). 27c4에 `sudo -n systemctl reboot` |
| 00:17:53 | 27c4가 내려갔다. head health는 200을 유지했다. |
| 00:18:21 | 워치독이 `gx10-27c4 dsv41-serve activating`(부팅 직후 27c4 자기 유닛이 기동 중)을 보고 head를 재기동했다(결정적 1/12). |
| 00:18:28 ~ 00:19:03 | head pre가 워커 3 대를 재기동하고 rank 0이 exec했다. |
| 00:19:30 ~ 00:21 | rank 3 `serve-node.sh`가 `mkdir: cannot create directory '/mnt/kvdisk/kv': Permission denied`로 끝났다. 20 s마다 5 회 재시도한 뒤 failed가 됐고, head는 dist init에서 rank 3을 기다렸다(health 000). |
| 00:29:09 | **사람 개입**: 27c4의 NFS 마운트를 DS4F `ensure_nfs_from_head`와 같은 옵션으로 되살렸다(`sudo -n mount -t nfs -o nfsvers=4.2,hard,timeo=50,nconnect=4,rsize=1048576,wsize=1048576,noatime 10.100.0.16:/mnt/kvdisk /mnt/kvdisk`). 재부팅 전 상태로 되돌린 것이다. |
| 00:29:24 | 워치독이 `gx10-27c4 dsv41-serve failed`를 보고 head를 재기동했다(결정적 2/12). dist init에서 멈춘 프로세스라 stop이 `TimeoutStopSec`까지 걸렸다. |
| 00:30:25 ~ 00:31:05 | head pre가 `reset-failed` 뒤 27c4, 37cc, f323을 재기동하고 rank 0이 exec했다. |
| 00:33:35 | health 200. 00:17:53부터 약 16 분 동안 중단됐다. |
| 00:33:41 | 웜업 3 요청 모두 ok(8K TTFT 9.10 s) |

- 원인: 워커의 `/mnt/kvdisk` NFS 마운트는 부팅 때 아무도 올리지 않는다(§5). 그런데 `serve-node.sh`가 모든 랭크에서 `mkdir -p "$KVFS_DIR"`를 실행해서, 마운트가 없는 워커는 root 소유 마운트포인트에서 실패했다.
- fs 티어는 rank 0 스케줄러만 쓴다. `vllm/v1/kv_offload/tiering/fs/manager.py:224`의 `makedirs`는 매니저 쪽 코드이고, 워커는 relay receiver라 `tiering/spec.py:402-429`에서 fs를 만들지 않는다.
- 조치: 커밋 `fix(dsv41): create the fs tier directory on rank 0 only`로 워커에서는 디렉터리를 만들지 않게 했다.
- 워치독과 head pre는 설계대로 동작했다. 두 번 모두 워커 상태를 보고 결정적 재기동을 걸었고, 마운트를 되살린 뒤로는 사람 없이 복구됐다.

#### 2차: 수정 반영 뒤, 통과(사람 개입 없음)

수정 커밋을 4 노드에 `git am`으로 반영했다. 27c4의 수동 NFS 마운트는 재부팅하면 사라지므로 실제 부팅 조건 그대로 시험했다.

| 시각 | 경과 | 사건 |
| --- | ---: | --- |
| 00:42:03 | | 가드를 확인했다(head active 658 s, health 200, 4 노드 active, 노드 트리가 수정 커밋). |
| 00:42:04 | 0 | 27c4에 `sudo -n systemctl reboot` |
| 00:42:23 | +19 s | 27c4가 내려갔다. head health 200 |
| 00:42:26 | | 27c4 부팅(`uptime -s`). `/mnt/kvdisk`는 마운트되지 않았다. |
| 00:42:49 | | 워치독: `unreachable: gx10-27c4 x1` |
| 00:42:59 | +55 s | 27c4 `dsv41-serve` active. 부팅 뒤 자기 유닛이 올라왔다. |
| 00:43:38 | +94 s | 워치독: `gx10-27c4 dsv41-serve restarted after the head` → head 재기동(결정적 3/12) |
| 00:43:56 | | head pre가 27c4를 재기동했다. rank 3 pre와 start에 오류가 없었다(수정 뒤 `Permission denied` 0건). |
| 00:44:31 | | rank 0 active |
| 00:47:06 | +302 s | health 200 |
| 00:47:21 ~ 23 | +333 s | 웜업 3 요청 모두 ok. 8K TTFT 19.92 s(27c4 재부팅 뒤 첫 요청이라 페이지 캐시가 차가웠다), sampled-thinking 0.38 s, image 0.48 s(답 `4`) |

- 복구 뒤 4 랭크 active, `NRestarts=0`이다. 워커 ActiveEnter(27c4 …037, 37cc …050, f323 …064)가 head(…071)보다 앞선다.
- 27c4의 `gpu-clock-cap`과 `earlyoom`은 active다. head pre의 캡 게이트도 통과했다.
- attention 파일은 없다. `restart-history`는 결정적 재기동 3건(1차 2건, 2차 1건)이다.

### d. head 재부팅: 통과(사람 개입 없음)

재부팅 전에 확인한 것:

- head `/etc/fstab`에 외장 SSD UUID로 `/mnt/kvdisk ext4 defaults,noatime,nofail`이 있다.
- `nfs-server`, `gpu-clock-cap`, `earlyoom`이 enabled다. linger yes, `dsv41-serve`와 `dsv41-watchdog.timer`도 enabled다.
- 00:48:39 가드: health 200, 4 노드 active

| 시각 | 사건 |
| --- | --- |
| 00:48:39 | head에 `sudo -n systemctl reboot` |
| 00:49:00 | head 부팅(`uptime -s`) |
| 00:49:09 | head pre 시작 |
| 00:49:12 | `GPU clock cap check failed (1/12)`. 부팅 직후라 cap 서비스 조회가 아직 준비되지 않았다. 10 s 뒤 재시도에서 통과했다. |
| 00:49:28 / 41 / 55 | 워커 27c4, 37cc, f323 재기동(각 워커 pre 1 s 이내) |
| 00:50:13 | rank 0 active |
| 00:52:3x | health 200. 재부팅 명령으로부터 약 4 분이다. |
| 00:52:34 ~ 36 | 웜업 3 요청 모두 ok. 8K TTFT 6.05 s, sampled-thinking 0.36 s, image 0.45 s(답 `4`) |

복구 뒤 상태:

- 4 랭크 active, `NRestarts=0`. 워커 ActiveEnter(…381, …394, …408)가 head(…413)보다 앞선다.
- `dsv41_ctl.sh caps`: 4 대 active, Enabled, 1989 MHz
- head: `/mnt/kvdisk`(/dev/sda1 ext4) 마운트, `nfs-server`, `earlyoom`, `gpu-clock-cap`, `dsv41-watchdog.timer`, `kvfs-gc.timer` 모두 active. MemAvailable 4.99 GiB, override 0 bytes, `Environment` 비어 있음, attention 파일 없음
- 워커 NFS: 37cc와 f323의 hard 마운트는 head 재부팅 뒤에도 응답했다(`stat /mnt/kvdisk/kv`). 27c4는 c에서 재부팅한 뒤 마운트되지 않은 채이며, rank 3 기동에는 영향이 없었다.

### e. 골든룰: 통과

| 항목 | 결과 |
| --- | --- |
| 기본값 | override 파일 0 bytes, `systemctl --user show dsv41-serve -p Environment` 비어 있음(4 노드) |
| health | 200 (01:02:35 `dsv41_ctl.sh status`) |
| caps | `dsv41_ctl.sh caps` 종료 코드 0, 4 대 active, Enabled, 1989 MHz(head 포함, 직접 조회) |
| 노드 트리 | 4 노드 dirty 0. HEAD는 `fix(dsv41): create the fs tier directory on rank 0 only`다. |
| 워치독 | `dsv41-watchdog.timer` active, attention 없음. 벤치 동안(01:00:05 ~ 01:02:19)은 타이머를 멈췄다. |
| 벤치 S128 | casebench solo 128K(`i32-golden-S128`, 16K 웜 뒤): 127,891 토큰 67.52 s, **1,894.2 tok/s**(기준선 1,792 ~ 1,836 이상. #17의 부팅 뒤 첫 긴 프리필은 1,807 ~ 1,913), head 4.85 → 최저 4.27 → 끝 7.20 GiB |
| 디코드 | `decodebench.py --types code --levels 1 --tag prod-i32`: 85.31 tok/s, `tokens_per_chunk` 5.82 → **스텝 68.2 ms**(프로덕션 68.1 ~ 68.5 ms), 수락 길이 불변 |
| head MemAvailable | 8.67 GiB (01:02:35, 128K 뒤 캐시가 풀린 상태) |

레코드는 head `~/sglang-cmp/results/casebench.jsonl`(`i32-golden-warm`, `i32-golden-S128`)과 `~/dsv41-prep/bench/decode-i32.jsonl`에 있다.

## 4. 구 DS4F 유닛

| 노드 | 유닛 | 조치 |
| --- | --- | --- |
| 6040 | `ds4f-log-capture.service` (system) | `sudo -n systemctl disable --now` → disabled, inactive |
| 37cc | `tp4-worker.service`, `kv-prune.timer` (user) | `disable --now` → disabled, inactive |
| 27c4 | `tp4-worker.service` (user) | `disable --now` → disabled, inactive |
| 6040 | `tp4-head`, `ds4f-head`, `ds4f-watchdog.timer`, `kv-prune.timer`, `kv-snapshot-ttl.timer` (user) | 이미 disabled. `serve-warmup.service`는 static |
| 27c4 | `ds4f-worker.service` (user) | 이미 disabled |

유닛 파일, `~/migration-027`, `~/ds4f-logs`는 그대로 두었다. `uvm-stall-sentinel.timer`(4 노드)와 `kvfs-gc.timer`(head)도 바꾸지 않았다.

## 5. 관찰

- 워커 3 대의 `/mnt/kvdisk` NFS 마운트는 런타임 마운트다(`mnt-kvdisk.mount` active). fstab, systemd 유닛, autofs 어디에도 정의가 없어, 구 DS4F의 pre(`ensure-kvdisk.sh`)가 올리던 것으로 보인다. 재부팅하면 돌아오지 않는다. V4.1F 워커 랭크는 fs 티어를 쓰지 않는다(`relay_from_rank0`, rank 3 프로세스가 `/mnt/kvdisk` 아래에 연 fd 0개).
