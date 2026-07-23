# 2026-07-23 채널-매트릭스 캘리브레이션 실측 데이터

밤샘 캠페인(1.5B/7B/14B TOFU + RWKU-7B, 3노드 24×H100, 260+ 셀)의 development
selection 진단값. 클러스터 셀렉터 출력(`runs/*/objective_freeze.recommended.yaml`)
에서 채팅 경유로 전사(transcribe)함 — 원본은 group-volume의 recommended.yaml이
단일 진실이며, 빈 셀은 전사 시점에 수치 미회수(판정은 셀렉터 출력 기준).

## 최종 판정 (선택 기준: recall≤0.10 ∧ mean ΔNLL≤2.0 ∧ CVaR05≤상한*)

| 캠페인 | output 계열 | representation 계열 | 상태 |
|---|---|---|---|
| TOFU 1.5B | 전멸 | 전멸 | 종결 — 공동 달성 불가 실측 (3R/25설정) |
| TOFU 7B | graddiff r4 마이크로 프로브 판정 대기 | ✅ rmu a10_c3_lr32_s240 | 진행 |
| RWKU 7B | 전멸 (4R — t000 mean vs t001 tail 양측 실패) | ✅ repnoise a1_c3_lr1_s480 (0.080/0.77/3.49, 원기준 8.0도 통과) | 종결 |
| TOFU 14B | ✅ graddiff lr1_s120 (0.038/0.46/1.69) · gru lr1_s120 (0.029/0.95/2.43) | 전멸 (rmu dose-flat 0.74→0.65→lr32도 실패) | 종결 |

*CVaR 상한: 원 8.0; 7B급 두 캠페인은 16.0 (prereg/AMENDMENT-2026-07-23-cvar-bound.md,
audit·프로브 결과 존재 전 채택, 양 기준 병행 보고).

## 파일

- `dev_diagnostics_{1p5b,7b_tofu,rwku_7b,14b}.csv` — 캠페인별 전 설정 진단
  (objective, setting, round, reach_max, mean_dnll, cvar05_dnll, verdict)
- `graddiff_frontier_per_target.csv` — 결정적 per-타깃 상세 (7B a198 꼬리 16.02
  vs 상한 16.0; RWKU 타깃별 반대 기준 실패; RWKU repnoise 해결 per-런)

## 논문(§5) 인용 포인트

1. **꼬리 집중**: 7B graddiff 도달 시 mean 1.31 vs CVaR 11.9 — 평균은 무해,
   꼬리가 파괴. 같은 설정에서 요청별 실패 축이 다름(확산형 vs 집중형).
2. **블록 용량 스케일 단조성**: 동일 계약에서 1.5B 불가 → 7B 꼬리 벽 → 14B 여유 통과.
3. **NPO/SimNPO 플래토 보편성**: 전 스케일·전 데이터셋에서 lr×16/스텝×4에도
   recall 0.35–0.86 정체, 손상 ~0.
4. **rmu 효능 스케일 역전**: 7B 0.78→0.21(가파름) vs 14B 0.74→0.65(평탄) —
   같은 조작이 스케일에 따라 반대로 반응.
5. **RepNoise 데이터셋 반전**: TOFU에선 도달 전 손상 폭발, RWKU에선 원기준 통과.
6. **Circuit Breakers 무반응**: 4세팅 × alpha 1–100 × lr 1–4e-6 전부 recall
   0.84–0.90 고정 — 이 계약에서 도달 불가.
7. **SFT 암기 예산의 스케일 의존**: probe_block 계약에서 1.5B=1200 / 7B=400 /
   14B=800 / RWKU-7B=200스텝 조기 도달(예산 800).

관련 커밋 재료: `docs/data/fidelity/`(7B bf16 게이트 실패 인증서),
`prereg/AMENDMENT-2026-07-23-cvar-bound.md`, `docs/plan_2026-07-23_fleet.md`(전 과정 기록).
