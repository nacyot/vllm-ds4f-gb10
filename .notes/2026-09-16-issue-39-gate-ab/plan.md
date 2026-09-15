# #39 같은 빌드 게이트 A/B 측정 계획

## 현재 동작과 문제

실행 정본은 #39 코멘트 `icmt-c6218ef6`이다. 이슈 본문의 새 브랜치·빌드·배포 요구는 이 코멘트가 대체했다. 현재 로컬 HEAD는 `d0f108eab6fc3fcecc73f5d1e3b2677045ced658`이며 작업 트리는 깨끗하다. 이번 계획 단계에서는 노드에 접속하거나 서비스를 조작하지 않았다. 실행 직전 네 노드의 같은 빌드 여부를 확인한다.

기존 `dsv41.env`는 prefault 비동기/청크를 1/8, 인덱서 TP 분할을 1로 채택했다. `dsv41_ctl.sh start`는 호출 환경의 KNOBS를 head override 파일로 쓰고 persistent unit을 재시작한다. `dsv41_unit.sh`가 워커 3대에 같은 override를 복사해 순서대로 재시작한다. `serve-node.sh`는 prefault 값을 engram config로 전달하고 분할 환경변수를 export한다. 인덱서는 랭크 간 게이트 일치를 검사한다. 따라서 게이트 변경은 재부팅으로 반영하며 네 랭크의 일치가 불변 조건이다.

이전 #37/#38은 개별 효과, #46은 채택 조합 전체 스위트를 측정했다. 아직 같은 빌드·같은 셀 순서에서 수정별 몫과 상호작용을 분리한 표가 없다. #46의 S128 하락도 긴 연속 측정 뒤의 순서 효과인지 회차 편차인지 남아 있다. 기존 결과 자료를 이번 통제 비교로 보완한다.

## 바꿀 것과 실행 접근 — 승인 이후

코드와 기본값은 그대로 두고 아래 네 번의 부팅에서 override만 바꾼다. 정상 경로에서 D가 최종 기본값 복구를 겸하므로 불필요한 다섯 번째 부팅은 하지 않는다.

| 순서 | 태그 | start에 전달할 knob |
| --- | --- | --- |
| A | i39-off | `ENGRAM_DECODE_ASYNC=0 ENGRAM_CHUNK_RUNS=1 DSV41_INDEXER_TP_SPLIT=0` |
| B | i39-pf | `ENGRAM_DECODE_ASYNC=1 ENGRAM_CHUNK_RUNS=8 DSV41_INDEXER_TP_SPLIT=0` |
| C | i39-split | `ENGRAM_DECODE_ASYNC=0 ENGRAM_CHUNK_RUNS=1 DSV41_INDEXER_TP_SPLIT=1` |
| D | i39-prod | 없음 — 네 노드 override 0 bytes |

1. 워크스테이션의 `/opt/homebrew/bin/bash deploy/gb10-cluster/dsv41/dsv41_ctl.sh`를 사용한다. 로컬과 네 노드의 HEAD·dirty·실행 파일을 읽기 전용으로 확인한다. 기준 빌드와 다르면 배포하거나 임의 비교하지 않고 불일치를 기록한다. 기존 `i39-*` 행·로그와 시작 행수를 확인해 태그 충돌 시 기존 결과를 덮거나 삭제하지 않는다.
2. 시작 전 caps 네 행 active/1989, head MemAvailable ≥5.2 GiB, metrics running/waiting=0, r0.log 마지막 200줄 POST /v1 출처가 127.0.0.1뿐인지 확인한다. 외부 요청이 있으면 10분 후 재확인하며 그동안 시작하지 않는다. 대기는 60초 이하 관측 단위로 나눈다. 만족하면 head watchdog timer를 stop하고 inactive 및 시각을 기록한다. 다른 타이머는 조작하지 않는다.
3. 호출 환경에 기존 KNOBS가 섞이지 않게 확인하고 각 표의 knob만 전달해 `start`한다. 별도 `stop`이나 직접 override 편집은 추가하지 않는다. start부터 health 200까지 시간을 기록하고 10초 간격, 최대 약 7분의 전경 폴링을 한다. 장시간 명령도 전경 세션으로 관측하고 30~60초 간격으로 진행 상황을 확인한다.
4. 네 노드 override의 내용 일치, 이번 부팅 구간 r0.log의 `mmap_decode_async`/`mmap_min_chunk_runs`, 네 랭크의 `DSV41_INDEXER_TP_SPLIT on` 줄 수(켜짐 각 1, 꺼짐 각 0)를 확인한다. 누적 로그 전체를 세지 않는다. 불일치 재시작도 아래 부팅 재시도 상한에 포함한다. 이번 부팅 warmup.log JSON 3줄 완료를 기다린 뒤 측정한다. 웜업이 끝나지 않으면 부하를 겹쳐 넣지 않고 이상으로 기록·복구한다.
5. head `~/sglang-cmp`에서 아래 셀을 순서대로, 셀마다 SSH 한 번씩 전경 실행한다. 실행 전 기존 casebench/decodebench/report 도구의 출력 필드와 append 관례를 확인하며 도구를 수정하지 않는다. `T`는 부팅 태그다.

```bash
cb() { python3 casebench.py --base http://127.0.0.1:8888 --config "$T" --out results/casebench.jsonl "$@"; }
```

| 셀 | 명령 |
| --- | --- |
| 1 warm | `cb --mode solo --tag "$T-warm" --prefill-tokens 16000` |
| 2 warm2 | `cb --mode mixed --tag "$T-warm2" --prefill-tokens 6000 --decodes 2` |
| 3 S128 r1 | `cb --mode solo --tag "$T-S128r1" --prefill-tokens 128000` |
| 4 S128 r2 | `cb --mode solo --tag "$T-S128r2" --prefill-tokens 128000` |
| 5 S128 r3 | `cb --mode solo --tag "$T-S128r3" --prefill-tokens 128000` |
| 6 S32 | `cb --mode solo --tag "$T-S32" --prefill-tokens 32000` |
| 7 decode | `python3 decodebench.py --base http://127.0.0.1:8888 --tag "$T" --types prose,code --levels 1,2,4 --gen 256 --out results/decode.jsonl` |
| 8 S128 r4 | `cb --mode solo --tag "$T-S128r4" --prefill-tokens 128000` |

decodeは1セルで6行を作る。全体は32セル、casebench 28行、decode 24行。見積もりは約50分で、待機・復旧時間は別記する。

1. 毎セル直前の head MemAvailable、`CELL_START/CELL_END`時刻、stdout/stderr、実際のrcを `results/i39.log` に残す。128K直前に3.0 GiB未満なら60秒ずつ3回待って再確認し、なお不足ならそのセルをskipして理由を記録する。終了時は新規JSONL行・health 200・ATTENTION不在を確認する。メモリ最小は既存サンプルとセル境界の観測最小として報告する。
2. D終了後は四ノード override 0 bytes、unit `Environment`空、health 200、caps四台active/1989、ATTENTIONなし、dirty 0、head MemAvailableを記録する。最後に head watchdog timerをstartしてactiveと再開時刻を確認する。

### 中断・復旧

- 7分以内にhealth 200にならなければCTL logを保存し、同じknobで1回だけ再試行する。再失敗ならknobなしstartで復旧して測定を終了する。
- セルrc≠0は記録して次セルへ。同一ブートで2セル連続失敗ならそのブートを打ち切り次ブートへ進む。skip・失敗・欠測を成功行に置き換えたり、予定外のセル再実行で埋めたりしない。
- health喪失やATTENTIONなど正常性の異常は追加負荷を止め、ログを保存して基本値復旧へ進む。ATTENTIONを削除して検査を通さない。
- 中断・SSH障害を含め、実験変更後の終了は必ずknobなしstart、ゴールデンルール確認、watchdog start/active確認まで行う。接続障害時は回復後に復旧を優先する。復旧不能なら未達条件を明記し、成功扱いしない。

## 影響サイトと保存範囲

- 実行時の影響先はhead gx10-6040とワーカーf323/37cc/27c4、既存:8888の利用者、serve/warmupユニット、watchdog。ブート中は提供が中断するため外部リクエスト不在を確認する。
- ゲートの消費者はEngram実装と `sparse_attn_indexer.py`。他の採択値 `SHORT_RESERVE=4096`、`SPEC_BLOCK_DROP=0`、`ENGRAM_RELEASE=0` を固定する。Aは過去の完全な巻き戻しではなく、この違いを明記する。
- 結果の消費者は既存 `report_tables.py`、#35/#39の効果判定と後続報告。JSONLスキーマと既存行を維持し、今回の行・ログ・ブート証拠のコピーをnotes内に保存する。ノードの既存resultsへの追記とoverride更新はレシピの実行時出力であり、リポジトリに含める変更は `.notes/2026-09-16-issue-39-gate-ab/` と `.gitignore` の `!.notes/2026-09-16-issue-39-gate-ab/` 一行だけとする。ignore例外の追加はimplementで行う。

## results.mdと検証方法

1. ブート表に開始時刻、healthまで秒、override内容、起動引数、splitログ数、観測メモリ最小を載せ、証拠ファイルをリンクする。
2. A/B/C/D表にS128 r1/r2/r3/r4 tok/s、S32、prose/code c1/c2/c4 per_stream tok/sとtokens_per_chunk、コードc1ステップms=`1000 × tokens_per_chunk / per_stream`を載せる。`python3 report_tables.py results i39-off i39-pf i39-split i39-prod`の出力も保存する。表ツールの集計・最初の一致行の選択だけに頼らず、個別タグを照合する。
3. 分割の寄与はC−AとD−B（S128 r2/r3平均、S32）。prefaultの寄与はB−AとD−C（コードc1ステップms、산문c1 tok/s）。絶対差と単位を示し、時間は負が改善、処理量は正が改善と明記する。条件ごとの効果が違えば観測上の相互作用として差を記載し、この回数で統計的確証とはしない。
4. 比較値は#38 off/on=1,666/1,836、#37 base=70.2/68.4とasync1=69.1/67.0（原資料の指標・単位を照合）、#46 S128=1,684/1,599/1,633とコードc1=86.0 tok/s、combo-nob12xコードc4合計208.6を添える。PR56562は既にrevert済みでゲートなし、#40の+0.3%は参考値とする。
5. 各構成の順序差=`r4 − (r2+r3)/2`。r1は初回長文プリフィルのペナルティを含むため主判定から除外する。4構成すべて同符号ならレシピ上の順序効果、符号が散れば回次偏差。ゼロ・欠測でこの二択が成立しなければ判定保留と明記する。
6. validateでは今回のappend区間とタグを照合し、casebenchは4構成×7タグ=28行、decodeは4構成×2種類×3並列度=24行を重複・欠落なく確認する。rc、実データのエラー、decode成功ストリーム数も検査する。skipや中断で不足した場合は不完全測定として報告し、完了基準通過とはしない。
7. 数式を原JSONLから再計算し、基本値復旧とwatchdog再開の証拠を確認する。差分は許可パスとignore一行のみ。コード変更がないため新規テスト・ビルド・ライブpytest/torchは行わず、既存ベンチと結果照合で検証する。

## 範囲外とplan-approveでのオーナー判断

新規ゲート、コード変更、ノードgit am、PR56562再導入、#46全スイート再測定、キャッシュ・KV結果の手動削除、キャップ操作、sysctl/swap操作、手動drop_caches、バックグラウンドチェーンは行わない。画面変更はない。

**判断が必要なのはstartの既存副作用を含む「削除なし」の範囲。** 現行 `dsv41_unit.sh` のpreは `shm_cleanup pre` の後に `sync; echo 3 > /proc/sys/vm/drop_caches` を実行する。`dsv41_lib.sh` のcleanupは所有者・実体パス・inode・未使用・サービス状態を検証した `/dev/shm` 候補を列挙し、個別にrmする。README運用規則はこのstart/stop内部の検証済み削除だけを認めている。一方、今回の要求は「削除なし」、SSOTも「どんなrm」「drop_caches」を禁止しつつ、同じstartを必須としている。

plan-approveでは、**禁止の範囲を手動追加操作とし、指定start内部の既存cleanup/drop_cachesは許容するか**を決めてもらう。許容されれば既存起動処理をそのまま使用し、対象・削除結果をログで保存する。安全確認が出たらユーザー指定の対象確認・監査記録手順に従う。内部副作用も禁止するなら、コード変更0かつ最後override 0 bytesという条件で現行startは実行できないため、起動前に要件の調整が必要。`DSV41_SHM_DRYRUN=1`はdrop_cachesを抑止せずoverrideも非空になるので、解決策として黙って追加しない。

Python実行器は今回ユーザーがSSOTと指定したコメンタリの明示例外に従い、遠隔の既存casebench/decodebench/reportに限って記載どおりsystem python3を用いる（#46も同じ扱い）。ローカルPython・依存環境インストールは不要。これは新たなオーナー選択肢とはしない。

この段階の成果物は本計画書のみ。提出でplan-approveへ進め、承認前にimplement・測定・サービス操作を行わない。
