# 이슈 #30 결과 — head 신규 부팅 여유 편차의 원인(웜업 컴팩션 회수의 anon/file 선택)과 493K 게이트 "Not run"

2026-09-13 00:35~00:43 KST, 워크트리 `issue-30-head-headroom`, 코드 5fd682ecdb(memlog 확장, head 에 6cdf193958 로 `git am`). 계획 `plan.md` §2 의 읽기 전용 진단(sar·journal·r0.log·#25 타임라인·VictoriaMetrics 15 s)에 더해 신규 부팅 1회를 1 s memlog(새 컬럼)로 실측했다.

## 1. 결론

- **편차의 정체**: 신규 부팅 유휴 여유 4.6~7.1 GiB 는 프로세스 크기·부팅 인자·코드 차이가 아니라, **첫 콜드 긴 프리필(82K 웜업)의 torch 할당자 재성장(2.93~2.95 GiB, 2 MiB 청크)이 단편화된 head free 풀에서 컴팩션·회수를 일으킬 때 커널이 파일 캐시를 회수하는가(→ 4.6~5.1) 콜드 anon 을 swap 으로 내보내는가(→ 7.1~7.8)** 로 갈린다. 세 프로세스 콜드 anon 합은 언제나 같다(worker 2,837 = RssAnon+VmSwap, EngineCore ~0.8 GiB, API ~0.9 GiB).
- **이번 부팅(00:37, 채택 구성)은 4.6~5.1 급**: 웜업 중 order≥9 블록 44 → 0, compact_stall +1,183(직접 컴팩션), kswapd 스캔 +516k 페이지, 회수는 Active(file) 1,436 → 88 MiB 의 파일 캐시에서, swap-out 3 MiB(pswpout +798), 3 프로세스 VmSwap 0. 웜업 뒤 유휴 5.11~5.21 GiB → `headroom` 5.2 미달(exit 3) → **493K 콜드 프리필 Not run**(레시피 4).
- **b5(7.1 급)와 #27 b2(7.7 급)의 swap-out 은 부팅이 아니라 웜업 프리필의 마지막 15 s** 에 났다(VictoriaMetrics 15 s: b5 19:33:45 swap 148 → 19:34:00 1,895 MiB; #27 b2 21:51:45 506 → 21:52:15 2,637 MiB). 웜업 외에 돌던 것은 없었다(bench2 는 b4 에서 끝, sentinel 로그 없음).
- SUnreclaim 은 4 노드 2,155~2,175 MiB 기준값(매니저 실측 + 이번 2,155~2,177). 이슈 본문 "AnonPages +1.7 GiB" 는 swap 배치 차이로 정정한다.

## 2. b5 타임라인 복원 (레시피 1) — plan.md §2-3 표 그대로. 요약

19:22 b4 부팅(채택) → 19:24 웜업(TTFT 53.9 s, 최저 3,340 MiB, swap 0) → 19:26 bench2 C1/C4(완료) → 19:29:21 stop → **19:30:21 b5 부팅** → 19:32:45 startup(부팅 인자·KV 4,571,060 토큰·그래프 0.86 GiB, #19 부팅과 diff 0) → 19:32:58 82K 웜업(TTFT 63.5 s; 다른 부팅 51~54 s) → **19:33:45~19:34:00 swap-out 1,747 MiB** + `2.93 GiB released` → 19:34:13 memlog: MemAvailable 7,114, MemFree 5,131, 파일 캐시 0.7 GiB, worker VmSwap 665 → 19:34:23 S4 493K(게이트 6.87, 첫 1 분 +0.5 GiB 추가 swap-out, 바닥 4.82).

## 3. 신규 부팅 1회 + 콜드 웜업 (레시피 2·3), memlog `~/dsv41-prep/bench/i30-boot-memlog.csv`(372 행, 00:36:37~00:42:47, 새 컬럼 포함)

| 시각(KST) | 단계 | MemAvailable | MemFree | 파일 캐시(Cached−Shmem) | order≥9 블록 | compact_stall 증분 | pswpout 증분 | 3 프로세스 VmSwap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 00:36:19 | stop 뒤(정지 12 s) | 122,444 | 12,739 | 2.6 GiB | 351 | 0 | 0 | — |
| 00:37:02 | 부팅 시작 직전 | 120,110 | 120,652 | 0.4 GiB | 14,957 | 0 | 0 | — |
| 00:38:0x | instanttensor 스트림 | 20,028 | 20,022 | 1.5 GiB | 773 | 0 | 0 | — |
| 00:39:14 / 00:39:41 | init engine(그래프 0.80 GiB) / health 200 | 4,912 | 3,592 | 2.3 GiB | 44 | 0 | 0 | — |
| **00:39:54 스냅샷 A** | 부팅 직후 | **4,965** | 3,646 | 2.3 GiB(Active 1,436 / Inactive 936) | 55 | 0 | 0 | W 0 / E 0 / A 0 |
| 00:40:04~08 | (실수) `W30` 은 기사용 salt → fs 티어 복원 750 MB, TTFT 1.7 s | 4,740 | 3,280 | | 34~40 | 0 | 0 | 0 |
| 00:40:27~00:41:26 | **콜드 82K 웜업 `Qk7`(87,614 토큰, TTFT 57.2 s, 정답)** | 최저 **2,993** | 최저 **1,540**(+29 s) | 6.3 → 4.7 GiB(Active(file) → 88 MiB) | 44 → **0**(5 s 만에) → 10 | **+1,183**(success +78) | **+798 페이지(3 MiB)** | 0 / 0 / 0 |
| 00:41:24 | `empty_cache … 2.95 GiB released` | 5,194 | 2,743 | | 10 | | | |
| **00:41:42 스냅샷 B** | 웜업 뒤 | **5,210** | 2,743 | 1.1 GiB | 10 | 1,183 | 798 | W 0 / E 0 / A 0 |
| 00:41:5x | `dsv41_ctl.sh headroom` | **5.11 GiB → exit 3** | | | | | | |
| 00:42:34 | 8K 프로브 `Rp4`(정답, TTFT 5.1 s) | 3,980 | 2,622 | | 10 | 0 | 0 | 0 |

3 프로세스 RssAnon(MiB): 부팅 직후 W 2,819 / E 805 / A 865 → 웜업 뒤 2,837 / 831 / 918 → 8K 뒤 2,838 / 830 / 918. VmSwap 전부 0. `MemorySwapPeak=0`. 작업 창 earlyoom 0. 웜업 중 pgscan_kswapd +516k, pgscan_direct +4,223, allocstall_normal +1,062.

해석: 부팅 자체는 회수 없이 끝나고(compact/pswpout 증분 0) order≥9 블록만 14,957 → 44 로 소진한다. 콜드 웜업의 2 MiB 청크 재성장이 곧바로 order≥9 = 0 에 부딪혀 직접 컴팩션 1,183 회·kswapd 회수를 일으킨다 — 여기까지는 b5 와 같은 경로다. 이번에는 커널이 **파일 캐시 ~1.6 GiB 를 회수하고 anon 은 두었다**(파일 LRU 2.3 GiB 가 있었고 swappiness 60 의 비용 균형이 파일 쪽이었음). b5 는 같은 자리에서 anon 1.75 GiB 를 내보내고 파일 캐시를 0.7 GiB 까지 비웠다(둘 다 회수). 어느 쪽을 택하는지는 그 순간의 파일 LRU 크기와 anon/file 재폴트 비용(직전 부팅들의 kvfs 복원·bench 로 인한 파일 churn)에 달린 커널 결정이라 부팅마다 다르며, 사용자 코드가 정하지 않는다. 이것이 4.6~7.1 편차의 전부다.

레시피 3(선행 작업 재현)은 별도 작업이 아니라 웜업 자체가 촉발점이라 이 부팅으로 갈음했다: 촉발(컴팩션·회수)은 재현됐고 anon swap-out 은 재현되지 않았다(수치 위).

## 4. 493K 콜드 프리필 게이트 (레시피 4) — Not run

시작값 5.11 GiB(웜업 뒤, `headroom` exit 3). 예상 바닥 = 5.11 − 2.1~2.2 = **2.9~3.0 GiB**(≥ 3.0 게이트 경계, earlyoom 2.43 위). 규칙대로 시작하지 않았다. README 표에 "Not run" 행 추가. 참고: 4.6~5.1 급에서 493K 를 돌리면 프리필 중 같은 컴팩션 경로가 anon 을 내보내 바닥이 예상보다 높을 수도(18:18 부팅의 493K 중 +1.15 GiB swap-out) 있으나 보장이 없어 가드 규칙은 유지.

## 5. 오너 레버 제안 (레시피 5) — 결정 이슈 등록

**#31** 로 등록(`ryno issue add`, 라벨 decision·dsv41·memory·kv500k): 선택지 (a) swappiness/MemoryHigh 로 콜드 anon 선방출, (b) 부팅 직후 compact_memory 1회, (c) EMPTY_CACHE 재성장 트레이드오프, (d) 현상 유지. 시스템 설정은 바꾸지 않았다.

## 6. 종료 상태

4 대 serve active, 캡 active·1989 MHz(부팅 순간 caps 표는 유휴 클럭 208~305 으로 찍히나 한도 2000 이하로 게이트 통과), health 200, head MemAvailable 3.98 GiB(8K 프로브 뒤, 정상 3.4~4.0), 작업 창 earlyoom 0, `i30-memlog` 유닛 정지, :8889 채택 구성 유지(이번 부팅이 곧 채택 구성).

## 7. 기록할 실수

- 첫 웜업 salt `W30` 이 kvfs 스토어에 이미 있어 콜드가 아닌 복원(750 MB)이 됐다. 새 salt `Qk7` 로 콜드 웜업을 다시 보냈고, 복원은 메모리 궤적에 영향이 없었다(order≥9 34~40, 컴팩션 0). 앞으로 웜업 salt 도 493K 처럼 "새 salt" 를 명시한다(README 반영).
- memlog 유닛 첫 기동을 상대경로로 해 즉시 종료 → 절대경로로 재기동(00:36:37, 부팅 전이라 손실 없음).
- 첫 커밋이 SPDX 헤더 훅에 막힌 상태에서 체인이 base 커밋(8c84efe16e, README 문서만)의 패치를 head 에 `git am` 했다(head 1e761cbbf3). 문서 동기화라 무해하며 head 트리는 clean.
