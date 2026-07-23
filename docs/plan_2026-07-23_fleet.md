# 2026-07-23 플릿 캠페인 플랜 — 장기 기억 (세션·노드 교체 시 이 문서부터)

> 마감: 페이퍼 **07-26 AoE (D-3)**. 오케스트레이션: `experiments/cluster/`
> (파일 큐, 노드당 GPU 8장 = 워커 8개). 런북: `docs/CLUSTER_FLEET_RUNBOOK.md`.

## 1. 지금 돌고 있는 것 — run261224-ul2 GPU0 (wave1 카나리아, 07-23 01:2x 시작)

- **큐**: `runs/cluster_queue/wave1` (유닛 3개: `fid__qwen25_7b`,
  `cal__qwen25_7b__a198`, `cal__qwen25_7b__a199`)
- **실행 중**: `cal__qwen25_7b__a198` — 7B fp32 calibration 그리드.
  이전 수동 런이 남긴 완료 셀 4개(ga×2, graddiff lr1/lr2)는 `--resume`이
  정확히 스킵했고 graddiff lr15_s120부터 이어가는 중. SFT 캐시 재사용
  (full-set mean NLL=0.608), fold 300 audit / 300 discovery 정상.

### 실험 목적 (칼리브레이션이 답하는 질문)

개발 전용 삭제요청(작가 198·199, audit 로스터와 불교차)에서 8개 언러닝
목적함수(ga, graddiff, npo, simnpo, gru, rmu, repnoise, circuit_breakers ×
lr/스텝 그리드)를 실제로 돌려 **"망각 기준(recall ≤ 0.10)에 도달하되 유지
지식이 붕괴하지 않는(mean ΔNLL ≤ 2.0, CVaR05 ≤ 8.0) 운영점"**을 목적함수마다
하나 고르는 것. 선택 기준은 예측력이 아니라 forget reach + utility만 —
답(예측 성능)을 보고 설정을 고르는 것을 차단하는 사전등록 구조.

### 예상 결과 (1.5B 실측 경험 기반 사전 예측 — 어긋나면 그 자체가 정보)

- **npo**: beta 0.1 필수 (1.0은 그래디언트 소멸로 reach 실패). lr 2~4e-6에서 도달 예상.
- **ga**: collapse 예상 (1.5B chanbal에서 `ga=collapsed` 플래그 전례) — stress 열로만 사용.
- **idkdpo / circuit_breakers**: notreached 가능성 (전례 있음) — CB는 240스텝 설정으로 재도전.
- **graddiff / rmu**: 무난히 도달 예상. rmu는 낮은 T2 기준선(1.5B에서 0.023 nats) 특성.
- **산출물**: 셀별 seal + damage.json → `select-freeze`가
  `objective_freeze.recommended.yaml` 추천 → 사람 검토 후
  `configs/channel_matrix/objective_freeze.yaml` **동결 커밋** (이게 wave2 게이트).
- **소요 실측 기입란**: a198 유닛 완료 시 셀당/유닛당 wall-clock 여기 기록 → 이후 웨이브 견적 근거.

## 2. 플릿 웨이브 설계 (수백 GPU, 노드 = 8×H100 단위)

원칙: **동결 게이트는 큐에 안 넣는다** (사람 단계). 독립적인 작업만 같은
큐에. 봉인(prereg) 위반이 되는 "남는 GPU 채우기"(audit 그리드 확장, 동결 전
alpha, audit 시드 추가)는 금지 — 남는 GPU는 아래 §3의 독립 트랙으로만 채운다.

| 웨이브 | 큐 | 유닛 | 의존성 | 예상 산출물/판정 |
|---|---|---|---|---|
| **W1** (진행 중) | `wave1` | fid 1 + cal 2 | 없음 | fidelity certificate PASS + 그리드 damage → freeze 추천 |
| W1.5 (CPU, 사람) | — | `select-freeze` → objective_freeze 커밋 → 클러스터 pull | W1 완료 | 목적함수별 단일 운영점 동결 |
| **W2** | `wave2` | `--phase audit` 3 (저자 181/186/191, 시드 2계열 내부 직렬) | W1.5 | 봉인 channel matrix: 7 objective × 7 predictor, Table 1(7B) 소재 |
| **W3** (W2와 병행 가능) | `wave3` | `--phase alpha-development` 2 (a198/a199×s2025) | W1.5 (audit 불필요) | model×parent α̂ 추천 (CVaR 최소화, recall≤.10 & utility≥.90 제약) |
| W3.5 (CPU, 사람) | — | `select-alpha-freeze` → alpha_protection_freeze 커밋 | W3 | α̂ 동결 |
| **W4** | `wave4` | `--phase alpha-audit` 6 (저자 3 × 시드 2) | W2? 아니오 — W3.5만. W2와 병행 가능 | 보호 endpoint: frozen α̂ vs no_repair/random/s0/s1, mean+CVaR95 |
| W5 (CPU) | — | aggregate, make_tables, evidence 파이프라인 | W2·W4 | Table 1/2 .tex + `evidence_readiness.json` |

주의: W2·W4는 `audit.offline: true` — 러너가 스스로 HF 오프라인 플래그를
켠다. sentence encoder 캐시가 없으면 audit이 실패하도록 설계돼 있으므로 W2
전에 각 실행 노드에서 `h100_campaign.sh prefetch` 1회 (공유 캐시라 1노드면 됨).

## 3. 남는 노드를 채우는 독립 트랙 (봉인과 무관, 언제든 가능)

우선순위순. 전부 커스텀 JSONL 유닛(런북 참조)이며 gate.py 계열은 run-tag
재사용 금지 특성상 `max_attempts: 1`.

- **T1 — Llama-3.1-8B 프로비저닝 + 제2 아키텍처 캠페인** (노드 1대 이상 흡수, 논문 일반성 최대 기여):
  `/group-volume/models/Llama-3.1-8B-Instruct` 다운로드 → config `enabled: true`
  커밋 → 7B와 동일한 W1→W4 체인을 별도 큐(`wave1_llama` 등)로. fidelity
  certificate 경로는 config에 이미 예약돼 있음.
- **T2 — 1.5B 시드 복제 (Table 1 CI 강화)**: chanbal2 재현 `--seed 2026, 2027, 2028...`
  새 run-tag로 GPU당 1런 (fp32 1.5B ≈ 58GB). 시드당 유닛 1개 — 노드 하나로 8시드.
- **T3 — 7B bf16 게이트 실패 아티팩트**: §5 주장("bf16은 게이트 실패") 뒷받침용
  `fd_fidelity.py` 7B bf16 1회. GPU 1장, 짧음.
- **T4 — xprot 기존 arm 재실행**: remote-quantile 버그 수정된 새 코드로
  crossed_protection 기존 arm 재측정 (기존 결과와 한 구현처럼 묶지 않기 위함 —
  Table 2를 메인에 쓸 경우에만 필요).
- **T5 — cost bench 재실측**: Table 4 수치(fd/jvp/vmap/streaming)를 새
  bench.py로 H100에서 재확인. GPU 1장, 짧음.

**규모의 정직한 한계**: 현 논문 스코프(7B+1.5B TOFU)는 수백 GPU를 상시
포화시키지 못한다. 캠페인 자체의 최대 동시폭은 W2+W3 병행 시점의 ~5유닛 +
독립 트랙. 수백 장을 진짜로 쓰는 길은 T1(모델 추가)과 W2 이후 페이퍼
evidence 계약의 타 벤치마크(WMDP/MUSE/RWKU/PISTOL — **어댑터 미구현**,
`src/rsus/data/registry.py`에 실제 어댑터를 등록해야 preflight가 열림)뿐.
마감(07-26) 전에는 T1까지가 현실적 상한.

## 4. 운영 체크리스트 (요약)

```bash
# 상태 확인 (아무 노드)
python experiments/cluster/workqueue.py status --queue runs/cluster_queue/<Q>
# 노드 투입 / 회수
bash experiments/cluster/launch_node.sh runs/cluster_queue/<Q>
pkill -f "experiments/cluster/worker.py --queue"
# 실패 triage는 런북 §실패 triage 절차 그대로 (partial dir 이동 후 retry-failed)
```

산출물 회수는 기존 규칙 유지: 수치·csv·json은 채팅으로 Claude 세션에 전달 →
세션이 재생성·커밋 (클러스터에서 git push 불가).
