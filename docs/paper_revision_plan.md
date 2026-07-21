# 논문 개편 계획 — 채널-조건부 스토리로 전면 재구성

상태: 계획 승인(2026-07-21). prereg는 `prereg/constants.yaml`의
`FREEZE-2026-07-21-channel-interaction` 블록으로 동결 완료 (chanbal/xprot3 언블라인딩 전).

---

## 1. 메인 테이블 재설계

### Table 1 (신규): 채널 × predictor-family 매트릭스
- 행 = predictor, **패밀리로 그룹** (gradient / representation / alignment-기각 / control),
  각 행에 backward-free 여부 열.
- 열 = objective, **선언 채널로 그룹**:
  output/loss-gradient (GA · GradDiff · NPO · SimNPO · IdkDPO · GRU) | representation (RMU · RepNoise · CB).
- 셀 = Spearman ρ. 열별 최고 굵게. 패밀리별 채널-평균 요약행.
- **캡션 headline = interaction Δ + bootstrap 95% CI** — leaderboard가 아니라
  "이기는 프로브가 채널에 따라 뒤바뀐다"가 메시지.
- 기존 표의 AUROC / Overlap@K / Tail ρ는 headline 프로브 2개(fd_norm, knn_feature)만
  Table 1b(또는 부록)로 이동.
- 생성: `experiments/paper/make_tables.py --report runs/<chanbal>/channel_report.csv`.

### Table 2 (신규): crossed protection (actionability)
- parent {GradDiff, RMU} × selector {none, random, fd_norm, knn_feature}.
- 열: forget recall, para recall, mean dNLL, **CVaR(5%)**; matched 셀 하이라이트.
- 성공 판정은 prereg의 protection_interaction 그대로.
- 생성: `experiments/paper/make_tables.py --crossed runs/xprot_<tag>/crossed.json`.

### Table 3: backward-free 정당화 (재배치)
- fd_norm fidelity (A/B/C, η=3e-3, R=64, ρ(A,C)=0.94) + 7B bf16 비용 실측
  (fd 5.97s/18.3GB vs vmap 10.0s/35.2GB vs streaming 17.4s/18.7GB).

## 2. 방법론 섹션 재구성 (Overleaf 이식용 뼈대)

1. **Setup** — d(x) ≈ g_x·Δθ; 부수피해 = 후보 민감도 × 업데이트 결합.
2. **Damage channels (신규 핵심)** — Δθ의 구조는 loss가 모델의 *어디*를 읽는가로 결정.
   output(토큰 우도/선호) vs representation(은닉 상태). 방법별 loss 식으로부터의
   **선언적** 분류표 (anti post-hoc; 코드의 `DECLARED_CHANNEL` 인용).
   이분법이 아닌 스펙트럼임을 명시 (GradDiff는 표현도 부수적으로 흔듦; RMU는 순수 표현).
3. **Predictor families + fd_norm** — 채널별 대응 프로브; fd_norm = K 랜덤방향
   central-diff 제곱평균의 ‖g‖² 추정(backward-free); A/B/C fidelity gate.
4. **Channel-matched protection** — profile → partition(P/R0) → guarded repair;
   selector 패밀리를 objective의 선언 채널에 매칭하는 라우팅 규칙.
5. **Protocol** — group-disjoint folds, sealed audit, matched forgetting, CVaR;
   **1차 지표 = interaction** (prereg freeze 인용).

## 3. 데이터셋 다각화 — 축 설계

프레임: **채널은 objective가 정하고, 데이터셋은 프로브의 base rate를 정한다.**
(forget–retain이 표면적으로 가까우면 lexical/embedding 프로브가 과대평가되는 식.)
따라서 데이터셋 축은 "interaction Δ > 0이 데이터 특성을 가로질러 성립"하는
강건성 축으로 판다. 각 데이터셋에 forget–retain 근접성 프로파일
(lexical overlap 분포, sentence-embedding 거리 분포)을 함께 리포트.

### 로스터 (prereg 동결됨)

| Tier | 데이터셋 | 지식 유형 | forget–retain 근접성 | group(폴드 단위) | 역할 |
|---|---|---|---|---|---|
| 주 | **TOFU** | 합성 entity QA | 높음(동형식 저자) | 저자 | 풀 매트릭스 + crossed + 복제 |
| 재현 | **WMDP-bio** (retain: MMLU 인접+원격 과목) | 위험 도메인 지식 | 낮음~중간 | MMLU 과목/토픽 | RMU 홈그라운드에서 interaction 재현 |
| 재현 | **MUSE-News** (BBC) | 실세계 코퍼스/사실 | 중간(동일 도메인 기사) | 기사/문서 | output 계열 홈그라운드에서 재현 |
| 재현 | **RWKU** | 실존 유명인 지식 | 중간(이웃 entity 내장) | entity | 사전학습 지식(비-SFT) 조건, 이웃-교란 audit 내장 |
| 스트레스 | **MUSE-Books** (Harry Potter) | 축자(verbatim) 기억 | — (축자 vs 일반능력) | 챕터/청크 | 축자 제거 조건에서 채널 구조 유지 여부 |
| 스트레스 | **PISTOL** | 합성 관계 그래프(계약) | **최고**(구조적 얽힘) | 계약/엔티티 쌍 | 근접성 극한에서 representation 프로브 과대평가 스트레스 테스트 |

커버 축: 합성↔실세계, entity↔코퍼스↔축자, SFT 주입 지식↔사전학습 지식,
근접성 低(WMDP)↔高(PISTOL), 무해↔위험(WMDP).

### 규모 규칙 (prereg의 reduced_roster_rule)
- 주(TOFU): objective 8종 × predictor 전체, crossed 포함, 시드 3종/다저자 복제.
- 재현/스트레스: **2×2 축소 roster** — objective {GradDiff, NPO | RMU, RepNoise},
  selector {none, random, fd_norm, knn_feature}. 1.5B fp32, GPU당 1런.
- 예상: 재현 1개 데이터셋 ≈ TOFU chanbal의 절반 이하 (objective 4종).

### 어댑터 구현 계획 (`src/rsus/data/`)
- 공통 인터페이스는 기존 `Request`/`CandidateUniverse` 그대로 — 데이터셋별
  `<name>_request()` 빌더만 추가 (tofu.py 패턴).
- 필요 요소: forget set, group 라벨이 있는 retain universe, (있으면) paraphrase/이웃 audit.
  - `wmdp.py`: forget = WMDP-bio 위험 QA, universe = MMLU 과목별 QA(과목=group),
    인접(생물 인접 과목) vs 원격(비과학) 구분을 native_audit에 반영.
  - `muse.py`: News/Books 공용 — 문서 청크를 QA-없는 seq-NLL 예제로; group=문서.
  - `rwku.py`: entity=group; 내장 neighbor set을 native_audit_ids로.
  - `pistol.py`: 계약 그래프에서 QA 생성; group=엔티티 쌍.
- 각 어댑터에 CPU 단위테스트(스키마/그룹 분리/manifest sha) — 컨테이너에서 검증 가능.
- HF 다운로드는 클러스터에서 1회(`HF_HOME=/group-volume/data/hf_home`), 이후 로컬 캐시.
  정확한 HF dataset id는 어댑터 작성 시 확인 (locuslab/TOFU 방식과 동일하게 고정 리비전 핀).

## 4. 실행 순서 (갱신)

| 순서 | 작업 | 담당 | 상태 |
|---|---|---|---|
| ① | prereg 동결 (interaction 기준 + 데이터셋 로스터) | Claude | **완료 (이 커밋)** |
| ② | Table 1/2 자동 생성 스크립트 + LaTeX 스켈레톤 | Claude | 진행 중 |
| ③ | 방법론 섹션 Overleaf 이식 초안 | Claude | 대기 |
| ④ | chanbal → Table 1/Fig 1, xprot3 → Table 2 확정 | Claude (런 종료 후) | 런 진행 중 |
| ⑤ | WMDP/MUSE/RWKU 어댑터 + CPU 테스트 → 클러스터 축소 런 | Claude(코드) + 사용자(GPU) | 대기 |
| ⑥ | PISTOL/MUSE-Books 스트레스, 시드/다저자 복제 | 〃 | 대기 |
