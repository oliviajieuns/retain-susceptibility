# Table 1/2 채우기 캠페인 — 플릿 설계 + 사전 예측 (2026-07-23)

> 목적: 논문 `tab:core-evidence`(Table 1)와 `tab:robustness`(Table 2)를
> 실측으로 채운다. 오케스트레이션은 `experiments/cluster/` 파일 큐
> (노드 = 8×H100 80GB, GPU 0–7 = 워커 8개). 점화 헬퍼:
> `experiments/cluster/enqueue_table12.sh`. 테이블 생성은
> `export_channel_matrix_raw.py → init_raw_plan.py → aggregate_raw.py →
> build_evidence.py --paper-root`가 전담한다.

## 1. 무엇이 어느 테이블 셀을 채우나

| 산출물 | 생산 웨이브/큐 | 소비자 |
|---|---|---|
| 7B TOFU audit 셀 (3저자×2시드, seal+traj) | `wave2` (`aud__qwen25_7b__a{181,186,191}`) | Table 1 패널 A (RQ1/RQ2), Table 2 primary 행 |
| 7B alpha-audit 셀 (frozen α̂ vs 4 comparator) | `wave4_alpha` (W3.5 α̂ 동결 후) | Table 1 패널 B (RQ3) |
| 14B audit + alpha | `wave1_14b` | Table 2 `Qwen2.5-14B` 행 |
| RWKU audit (fd_fidelity `--dataset rwku` 인증 후) | `wave_rwku` | Table 2 `RWKU` 행 |
| WMDP W1(fidelity+calibration)→freeze→audit | `wave_wmdp` | Table 2 `WMDP-bio/MMLU` 행 |
| Llama-3.1-8B 프로비저닝→W1→…| `wave_llama` | Table 2 `Llama-3.1-8B` 행 |
| 1.5B: 추가 GPU 불필요 | (종결) | Table 2 boundary 행 — 도달불가 실측으로 채움 |
| MUSE-News / PISTOL | 실행 불가 (캐시·망 부재) | Table 2 planned/not-run 행 (denominator 유지) |

파이프라인 (CPU, 클러스터 로그인 노드 또는 세션):

```bash
# 1) 개발 셀렉션 동결(selection_freeze) 후 불변 플랜
python experiments/paper/init_raw_plan.py --selection-freeze configs/paper/selection_freeze.yaml \
  --setting tofu_qwen25_7b --out results/paper/raw_plan.json
# 2) run 산출물 → 후보 레벨 raw shard (+ fidelity 요약)
python experiments/paper/export_channel_matrix_raw.py \
  --campaign-config configs/channel_matrix/7b_tofu.yaml --setting-id tofu_qwen25_7b \
  --prediction-alpha-freeze configs/paper/selection_freeze.yaml \
  --control-predictor knn_embed \
  --fidelity-certificate runs/channel_matrix_7b/fidelity/qwen25_7b.json \
  --out-dir results/paper/raw/tofu_qwen25_7b
# 3) 정규화 렛저
python experiments/paper/aggregate_raw.py --plan results/paper/raw_plan.json \
  --prediction-raw results/paper/raw/tofu_qwen25_7b/prediction.jsonl \
  --protection-raw results/paper/raw/tofu_qwen25_7b/protection.jsonl \
  --out results/paper/evidence_ledger.json
# 4) Table 1/2 .tex + 헤드라인 매크로 (paper 레포 루트를 --paper-root로)
python experiments/paper/build_evidence.py --paper-root ../paper
#    → sections/generated/table_core_evidence.tex / table_robustness.tex
```

## 2. 노드 배치 (수십 GPU 규모, 노드 = 8×H100)

우선순위순. 총 소요는 audit 유닛당 수 시간(7B fp32, GPU 1장 1유닛) 기준.

| 노드 | 큐 | 유닛 수 | 비고 |
|---|---|---|---|
| N1 | `wave2` | 7B audit 3 + alpha-dev 2 → (α̂ 동결 후) alpha-audit 6 | **Table 1 크리티컬 패스** — 가장 먼저 |
| N2 | `wave1_14b` | 14B audit 3 + alpha 계열 | Table 2 모델-스케일 행 |
| N3 | `wave_wmdp` | WMDP fid 1 + cal 2 (이후 라운드 추가) | 캘리브레이션 라운드가 병목 (7B 전례: 3–4라운드) |
| N4 | `wave_llama` | provision → fid 1 + cal 18 | `provision_llama.sh` 선행 (HF 미러 필요; 게이트 레포 토큰 필수) |
| N5 (GPU 1장) | `wave_rwku` | fd_fidelity `--dataset rwku` 1 → audit 3 | 인증서만 나오면 짧음 |
| 여유 노드 | T2 시드 복제 (`make_replicate_units.py`) | 8/노드 | Table 1 CI 강화용 (audit 시드 2025-2026 완료 후에만 의미) |

운영 수칙 (런북 준수): 띄우기 전 `nvidia-smi` 빈 GPU 확인, 로그는
유닛 러너가 호스트명 포함 파일로 자동 분리, run-tag 재사용 금지,
freeze 커밋 전 audit 점화 금지 (`enqueue_table12.sh`가 grep으로 강제).

## 3. 사전 예측 — 현재 실측으로 본 Table 1/2 전망

근거: `docs/data/calibration_2026-07-23/`(개발 진단), chanbal2(1.5B 예측
실측), xprot/알파 사전 실험(1.5B 수리 실측), 7B bf16 게이트 인증서.
아래는 **예측이지 결과가 아니다** — audit은 봉인돼 있고, 어긋나면 그
자체가 §5 보고 소재다.

### Table 1 (7B TOFU, parent 7행)

| Parent | 예상 E (eligible) | RQ1 pass 전망 | RQ3 pass 전망 | 근거 |
|---|---|---|---|---|
| GradDiff | **y** (해결 코어, lr08_s240) | **중~높음** | **중간** | 1.5B에서 grad_norm rho 0.489/fd_norm 0.400; 7B 데미지 꼬리 집중(mean 1.31 vs CVaR 11.9) → tail lift 유리. 수리도 1.5B에서 -2.6% 재현. 위험: a198 슬랙 0.41(15.59/16.0)로 시드 2026에서 non-reach 가능 |
| RMU | **y** (해결 코어, lr32_s240) | **중간** | **낮~중** | representation 채널은 knn_feature 우세(chanbal interaction CI>0). 위험: rmu 데미지 플로어가 낮으면(1.5B에서 0.023nats) 손상 분산 부족 → 부트스트랩 페어드 게인 CI가 0을 걸칠 수 있고, 수리할 것 자체가 적어 RQ3 8-UCB 동시 통과가 어려움. "diffuse damage → random wins" 실패 경계 후보 1순위 |
| NPO/SimNPO | n (recall 플래토 0.66–0.86, 도달 실패) | -- | -- | non-reach 행으로 보고 (denominator 유지) |
| GRU | n (도달 전 손상 폭발) | -- | -- | 〃 |
| RepNoise | n (TOFU에서 noise 항 지배) | -- | -- | 〃 (RWKU와의 데이터셋 반전이 §5 소재) |
| CB | n (recall 0.84–0.90 무반응) | -- | -- | 〃 |

- **RQ2**: fd 기계의 수치 무죄는 fp32에서 반복 확인(ρ(A,C) 높음, bf16만
  실패). f_ρ/f_K 플로어(0.80/0.70) 통과는 높은 확률. 단 **g_ctl(최강
  단순 컨트롤 대비 이득)**은 1.5B에서 fd_norm이 grad_norm 예측력의
  ~80%였음을 감안하면 컨트롤이 knn_embed일 때 통과 가능성 중간.
- **핵심 리스크**: audit 시드 2개×저자 3개 = 부트스트랩 지지 6유닛.
  IUT(RQ1 4멤버, RQ3 12멤버)의 동시 통과는 효과가 클 때만 가능. 통과
  실패 시에도 Table 1은 bound와 E/P로 정직하게 채워진다.

### Table 2 (9행)

| 행 | 예상 채움 | 전망 |
|---|---|---|
| held-out TOFU (7B) | 완전 | 위 Table 1과 동일; graddiff+rmu 중 하나라도 양 클레임 통과 시 full-chain 지원 |
| WMDP-bio/MMLU | W1부터 신규 | **미지수 최대** — 캘리브레이션 3–4라운드 소요 전례상 마감(07-26) 내 audit 도달이 빠듯. cal이 안 끝나면 planned/cal-only로 보고 |
| MUSE-News | planned/not-run | 데이터 부재 (캐시 없음 + HF 차단) — denominator 행 |
| RWKU | audit 3셀 | repnoise만 코어 → representation 쪽만 지원. **output 계열 전멸 실측 → full-chain 불가 확정적** |
| MUSE-Books (stress) | planned/not-run | knowmem만 캐시 (부분) — stress라 breadth에 영향 없음 |
| PISTOL (stress) | planned/not-run | 〃 |
| Qwen2.5-1.5B (boundary) | 도달불가 실측 | non-reach 7행 — 블록 용량 스케일 경계 서사 |
| Qwen2.5-14B | audit 3셀 | graddiff·gru 코어(output만) → representation 쪽 실패. full-chain 불가, 그러나 output 행 RQ1/RQ3은 시도 가능 (14B는 손상 경계 여유: cvar 1.69) |
| Llama-3.1-8B | 프로비저닝 성공 시 W1 | 마감 내 cal까지가 현실적 상한 (T1 전례) |

### 페이퍼 레벨 breadth 판정 (multi_setting_rule v2)

- 요구: primary(7B) full-chain ∧ 모델 그룹(14B/Llama) ≥1 ∧ 데이터셋
  그룹(WMDP/MUSE-News/RWKU) ≥2.
- **현 실측 기준 전망: NOT LICENSED가 유력.** RWKU는 output 전멸,
  MUSE-News는 실행 불가라 데이터셋 그룹 ≥2가 사실상 막혀 있고, 14B는
  representation 전멸이라 모델 그룹도 Llama에 달려 있다.
- 이는 프레임워크 실패가 아니라 **계약이 설계대로 작동하는 것**이다.
  §6.5의 서사("boundaries determine the licensed claim") 그대로:
  Table 2는 실패 경계(반전·플래토·도달불가)를 실측으로 채우고,
  TransferHeadline은 not-licensed를 명시 보고한다. 채널-반전 관찰
  (RepNoise TOFU↔RWKU, rmu 7B↔14B)이 §5의 실질 기여로 남는다.

## 4. 산출물 회수 규칙 (기존 유지)

audit/alpha 완료 후 `export → aggregate → build_evidence`는 CPU 단계라
클러스터 로그인 노드에서 실행 가능. 생성된
`results/paper/evidence_ledger.json`·`evidence_readiness.json`·두
테이블 .tex는 채팅으로 세션에 전달 → 세션이 재생성·커밋 (클러스터 push
불가). `--require-ready`가 0이 아닌 동안은 테이블에 \tblph가 남아 있는
정상 상태다.
