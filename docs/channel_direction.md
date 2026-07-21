# Retain Susceptibility — 방향 전환: Channel-conditioned Susceptibility

> **한 줄 요약**
> 유지(retain) 행동의 취약성은 **모델 고유의 한 숫자가 아니라, 언러닝 목적함수가 모델을 손상시키는 "채널"에 조건부**다.
> 우리는 이 채널을 사전에(언러닝 전) 싸게 예측하고, **채널에 맞춘 보호로 어떤 언러너든 부수 피해(특히 꼬리)를 무-망각-비용으로 줄인다.**

---

## 1. 페이퍼 방향성 (먼저 못박기)

### ❌ 이렇게 안 간다
> "우리가 최고의 언러너다 / SoTA 언러닝을 이긴다"

- SoTA 언러닝은 벤치마크마다 다르고 계속 바뀌는 표적 → "다 이긴다"는 리뷰어가 **반례 하나**만 찾으면 무너짐.
- 우리 기여는 언러닝 목적함수(NPO/RMU…)와 **경쟁하는 게 아니라 그 위에 얹는 예측+보호 레이어** → 직교적/보완적.
- 실측 이득도 완만함 (예: npo 1.146 → 1.097, CVaR ~5%). "SoTA 압살"로 팔면 5%가 초라해 보임.

### ✅ 이렇게 간다
> "우리는 **어떤 언러너든** 채널-매칭으로 더 안전하게 만든다 — 특히 **꼬리(worst-case)**에서, **공짜로(무-망각-비용)**."

- **성공 판정 = crossed 실험의 interaction** (matched > mismatched / random / none). **절대 리더보드 1등이 아님.**
- 5%가 작아 보이면 → **4박자로 판다:**
  1. **Tail (CVaR)** — 평균이 아니라 최악-꼬리 부수 피해를 줄인다.
  2. **무-망각-비용** — 같은 망각 수준에서 부수 피해만 줄인다.
  3. **Backward-free 저비용** — 후보별 역전파 없이(forward만) 예측한다.
  4. **채널-매칭 필수성** — 채널을 안 맞추면(mismatched) 이득이 사라진다 → 프레임 자체가 정당화됨.

> 💡 SoTA를 이기는 비교는 **나중에 부가적으로** 들어갈 수 있지만, 논문의 뼈대는 "**메커니즘을 아름답게 설명**"하는 것.

---

## 2. 왜 채널을 나누는가 (output vs representation)

부수 피해는 1차 근사로 `d(x) ≈ g_x · Δθ` — **후보의 민감도(g_x)와 업데이트(Δθ)의 결합**이다.
Δθ가 어떻게 생겼는지는 **목적함수의 loss가 모델의 "어디"를 읽어서 계산되는가**에 달려 있다. 그래서 채널은 loss가 읽는 위치로 갈린다:

| 채널 | loss가 읽는 곳 | 대표 방법 (loss 계산식) | damage를 지배하는 후보 성질 |
|---|---|---|---|
| **Output / loss-gradient** | 모델 **출력** (토큰 우도·선호) | GA, GradDiff, NPO, SimNPO, IdkDPO, GRU, KL-매칭 | **그래디언트 크기** ‖∇ℓ_x‖ |
| **Representation** | 모델 **내부 표현** (은닉 상태·레이어 활성값) | RMU, RepNoise, Circuit Breakers | **표현 근접성** (forget과의 은닉공간 거리) |

- **Output 채널**: "금지된 답의 확률을 낮춰라" → 같이 망가지는 건 그 파라미터에 **민감한(그래디언트 큰)** 후보.
- **Representation 채널**: "내부 표현을 랜덤/직교 방향으로 밀어라" → 같이 망가지는 건 표현공간에서 **가까운** 후보.

> **"신경망 loss가 두 가지만 읽는다"는 법칙이 아니다.** 원리적으로 loss는 아무거나(어텐션, 가중치, 그래디언트…) 읽을 수 있다.
> 다만 **현존하는 언러닝 방법은 사실상 출력 아니면 표현을 읽는다.** 우리 2채널은 "실제 방법들의 경험적 분류"이고, 각 방법의 **정의에서 바로 읽힌다**(결과 보고 정하는 사후 라벨링 아님). 새 위치를 읽는 방법이 나오면 프레임은 그 채널로 확장된다.

**핵심 정직함**: 엄격한 이분법이 아니라 두 극(pole)의 스펙트럼이다. GradDiff 같은 출력 방법도 **부수적으로 표현을 흔들어서** representation 프로브가 부분적으로 예측한다. 반면 **RMU는 순수 표현이라 gradient 프로브가 완전히 눈먼다** — 이 비대칭이 가장 깨끗한 증거다.

---

## 3. 메인 그림 읽는 법

### (A) 히트맵 — 세로축(predictor) × 가로축(objective/channel)

**세로축 = susceptibility 프로브(예측기), 패밀리로 묶임:**

| 프로브 | 무엇을 재나 | 비용 |
|---|---|---|
| **grad_norm** | 후보 자신의 손실 그래디언트 크기 ‖∇ℓ_x‖ (파라미터 민감도) | 정확하지만 **후보별 역전파**(비쌈) |
| **fd_norm** (우리 것) | grad_norm의 **backward-free 추정** — 랜덤 방향 ±η 교란한 forward 두 번의 제곱평균으로 ‖∇ℓ‖² 추정 | **후보 역전파 없음**(쌈) |
| **knn_feature** | 모델 **은닉 상태** 공간에서 forget과의 근접성 (표현 유사도) | forward만 |
| knn_embed | 외부 문장 임베딩 공간 근접성 | forward만 |
| knn_lexical | 답변 토큰 겹침 (어휘 유사도) | 모델 불필요 |
| fd (정렬) | forget 상승 **방향과의 정렬**(방향미분) — **기각된 접근** | forward만 |
| random_rank | 무작위 (바닥/대조군) | — |

**가로축 = 언러닝 목적함수, 채널로 묶임** (output 채널 | representation 채널). 셀 = 그 프로브가 그 objective의 실제 damage를 얼마나 잘 예측하나 (Spearman ρ).

**보이는 것 (7B, tofu-a180 예비 결과):**
- **gradient 패밀리**(grad_norm, fd_norm): output objective(GradDiff)에서 밝음(ρ≈0.3~0.36) → **RMU에서 꺼짐/음수**(≈0, −0.12) = **표현 채널에 눈멂**.
- **representation 패밀리**(knn_*): **RMU에서 가장 밝음**(ρ≈0.5), output objective에서도 부분적.
- **fd(정렬)**: 전 열에서 음수/무의미 = 실패 (수치는 정확한데 재는 양이 틀림).

### (B) Crossover 선그림 — "채널마다 이기는 프로브가 다르다"

두 헤드라인 프로브의 **채널별 평균 ρ**:
- 🟠 **주황 = fd_norm (gradient 프로브)**: output 채널에서 높고 → representation 채널에서 **내려감** (예: 0.24 → 0.12)
- 🟢 **청록 = knn_feature (representation 프로브)**: output 채널에서 중간 → representation 채널에서 **올라감** (예: 0.23 → 0.55)

**두 선이 교차(crossover)한다 = 이기는 프로브가 채널에 따라 뒤바뀐다.**
특히 **오른쪽 끝의 큰 격차(청록 0.55 vs 주황 0.12)** = gradient 프로브가 representation 채널에서 눈머는, 가장 선명한 증거.

> 색 주의: 히트맵의 빨강/파랑(양/음 상관)과 헷갈리지 않도록, crossover는 **범주형 색(주황/청록, colorblind-safe)**을 씀. 여기서 파랑(청록)은 "나쁨"이 아니라 그냥 다른 프로브임.

### 이 그림들의 의미

**어느 프로브도 항상 이기지 않는다.** → susceptibility는 모델 고유의 한 숫자가 아니라 **채널 조건부**. 올바른 예측기는 언러닝 목적함수의 damage 채널에 맞춰야 한다. (이것이 기존 단일-predictor 접근이 들쭉날쭉했던 이유 — 채널-blind였기 때문.)

---

## 4. 우리 기여 (3가지)

1. **채널-조건부 susceptibility 규명** — 취약성은 objective가 유도하는 damage 채널에 따라 다른 predictor family로 예측된다(실증).
2. **gradient 채널용 backward-free 프로브 (fd_norm)** — 후보별 역전파 없이 forward만으로 그래디언트 크기를 추정, exact grad_norm의 ~80~90% 예측력.
3. **채널-매칭 표적 보호** — objective 채널에 맞춘 selector로 보호 예산을 배분해, 무-망각-비용으로 부수 피해(특히 꼬리)를 줄인다.

---

## 5. 성공 판정 = crossed protection 실험

리더보드가 아니라 **interaction(교차 개입)**으로 판정한다:

| parent 언러닝 | none | random | fd_norm (gradient selector) | knn_feature (representation selector) |
|---|---|---|---|---|
| **GradDiff** (output 채널) | 기준 | ~기준 | **matched → 최저 damage 기대** | mismatched |
| **RMU** (representation 채널) | 기준 | ~기준 | mismatched | **matched → 최저 damage 기대** |

- 지표 = audit 부수 피해의 **평균 + CVaR(꼬리)**, **matched 망각 수준**에서.
- **성공 = 각 parent에서 matched의 CVaR이 mismatched/random/none보다 낮음.**
- 이게 나오면: "채널 매칭이 실제 보호를 개선한다 = 예측이 **actionable**" — 논문의 payoff.
- 추가로 selector에 grad_norm(비쌈)도 넣어 **fd_norm(쌈)이 grad_norm만큼 보호하는지** 검증 → "backward-free 저비용" 박자 확정.

---

## 6. 현재 상태 & 다음 단계

**확립됨**
- fd 정렬 프로브 기각 (jvp로 수치 무죄 확인, 실제 damage와 음/무상관).
- fd_norm fidelity 통과 — η=3e-3, R=64에서 grad_norm의 충실한 backward-free 추정 (ρ≈0.93).
- 두 predictor 패밀리가 서로 다른 후보를 고름(교차 상관 ≈0), 채널-매칭 prediction 예비 확인.

**진행 중**
- **채널 매트릭스 재실행**: output 5개(GA/GradDiff/SimNPO/IdkDPO/GRU) × representation 3개(RMU/RepNoise/Circuit Breakers) — 균형 잡힌 채널.
- **crossed protection**: (GradDiff, RMU) × {none, random, fd_norm, knn_feature} — interaction 검증.

**해야 할 것**
- multi-author 복제(단일 요청 → 여러 저자에서 블록·interaction 반복).
- (선택) 새 representation 방법(RepNoise/CB) 예산 캘리브레이션.
- (선택, 나중) 외부 보호 베이스라인과 정면 비교.

---

*문서 상태: 방향 전환 제안 (예비 7B 단일-요청 결과 기반). 숫자는 재실행/복제로 확정 예정.*
