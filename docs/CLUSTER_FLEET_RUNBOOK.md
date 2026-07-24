# 멀티노드 H100 플릿 런북 (노드당 8×H100, 총 100장+)

`experiments/cluster/`는 스케줄러 없는 사내 클러스터에서 8-GPU 노드 여러 대를
하나의 작업 풀로 묶는 최소 오케스트레이션 계층이다. 조율 매체는 전 노드가
공유하는 `/group-volume` 레포 안의 파일 큐 하나뿐이다 — 데몬도, 추가 의존성도
없다 (표준 라이브러리 + pyyaml).

```
experiments/cluster/
  workqueue.py     공유 파일 큐 (pending → claimed → done|failed, 원자적 rename)
  worker.py        GPU 1장을 전담하는 워커 루프 (하트비트 + 로그 + 재시도)
  make_units.py    캠페인 config에서 작업 단위(JSONL) 생성/즉시 enqueue
  launch_node.sh   노드 부팅: venv 활성화 후 GPU당 워커 1개 nohup 기동
  next_actions.py  읽기 전용 오라클: 지금 enqueue 허용된 페이즈/막힌 이유를
                   동결 게이트 기준으로 보고 (자율 에이전트는 이걸 먼저 실행;
                   레포 루트 AGENTS.md가 에이전트용 운영 계약)
```

## 핵심 설계

- **작업 단위 = 기존 러너가 이미 지원하는 최소 샤드.** run 디렉토리가 단위 간
  절대 겹치지 않도록 자름: calibration/audit은 `--only-authors <한 명>`,
  alpha 페이즈는 `--worker --author A --seed S`. 모든 명령에 `--resume`이
  들어가므로 재시도/스테일 회수가 안전하다.
- **GPU당 워커 1개.** fp32 7B는 H100 한 장을 거의 다 쓰므로 겹배치 금지 규칙을
  코드로 강제 — 워커는 시작 시 `nvidia-smi`로 자기 GPU에 1GiB 이상 상주 메모리가
  있으면 기동 거부(`--allow-busy-gpu`로만 해제).
- **크래시 복구.** 워커는 60초마다 하트비트 파일을 갱신한다. 노드가 죽으면
  `workqueue.py requeue-stale`이 오래된 claim을 pending으로 되돌린다(시도 횟수
  증가, 기본 `max_attempts=2` 소진 시 failed로 이동).
- **동결 경계는 큐에 넣지 않는다.** select-freeze / select-alpha-freeze는
  사람 리뷰 단계 그대로: 한 페이즈를 다 비우고 → 셀렉터 실행 → freeze 커밋 →
  다음 페이즈 enqueue.

## 사용 순서

### 0. 매 세션 공통 (기존 규칙 그대로)

```bash
source /group-volume/jieuns.shin/venvs/exp/bin/activate
cd /group-volume/jieuns.shin/retain-susceptibility
git pull && python -m pytest -q
```

### 1. 작업 enqueue (아무 노드에서 1회)

```bash
# calibration 페이즈를 큐에 적재 (모델×저자 샤드)
python experiments/cluster/make_units.py \
  --config configs/channel_matrix/7b_tofu.yaml \
  --phase calibration --enqueue --queue runs/cluster_queue/calib

# 여러 페이즈를 한 큐에 순서대로 쌓을 수도 있다 (의존성 없는 것끼리만!)
python experiments/cluster/make_units.py \
  --phase fidelity --phase calibration \
  --enqueue --queue runs/cluster_queue/wave1
```

`--out units.jsonl`로 파일만 뽑아 검토 후 `workqueue.py enqueue`로 넣어도 된다.
같은 unit id는 큐 어느 상태에 있든 재적재 거부 (seal append-only 원칙과 동일).

### 2. 노드 투입 (노드마다 1줄)

```bash
# 8-GPU 노드 전체 투입 — 워커 8개가 nohup으로 뜨고 즉시 반환
bash experiments/cluster/launch_node.sh runs/cluster_queue/calib

# GPU 일부만 쓰거나, 큐가 비면 워커가 스스로 종료하게 하려면
bash experiments/cluster/launch_node.sh runs/cluster_queue/calib 4
WAIT=0 bash experiments/cluster/launch_node.sh runs/cluster_queue/calib
```

기본값 `WAIT=1`이면 큐가 비어도 워커가 30초 간격으로 폴링하며 대기하므로,
**노드를 먼저 다 띄워놓고 나중에 페이즈를 enqueue하는 운영이 가능**하다.
13개 노드에 같은 명령을 치면 워커 ~104개가 한 큐를 나눠 가진다.

### 3. 모니터링 / 정비

```bash
python experiments/cluster/workqueue.py status --queue runs/cluster_queue/calib
# → 상태별 개수 + 실행 중 unit의 host/gpu/하트비트 나이 + 실패 unit의 exit code/로그 경로

# 죽은 노드의 claim 회수 (하트비트 30분 초과 기본)
python experiments/cluster/workqueue.py requeue-stale --queue runs/cluster_queue/calib

# 원인 수정 후 failed 전체 재시도
python experiments/cluster/workqueue.py retry-failed --queue runs/cluster_queue/calib

# 노드 하나의 워커 전부 중지
pkill -f "experiments/cluster/worker.py --queue"
```

로그는 unit 단위로 `runs/logs/cluster/<unit>__<host>_gpu<g>__try<n>.out`,
워커 자체 로그는 `runs/logs/cluster/worker_<host>_gpu<g>.out` — 호스트명이
파일명에 박히므로 공유 볼륨에서 덮어쓰기 사고가 없다.

## 캠페인 페이즈 → 큐 웨이브 매핑

| 웨이브 | enqueue | 샤드 수(모델당) | 완료 후 사람 단계 |
|---|---|---|---|
| 1 | `--phase fidelity --phase calibration` | 1 + 2 | `h100_campaign.sh select-freeze` → objective_freeze 커밋 |
| 2 | `--phase audit` | 3 | `h100_campaign.sh aggregate` |
| 3 | `--phase alpha-development` | 2 (저자×시드) | `select-alpha-freeze` → alpha_protection_freeze 커밋 |
| 4 | `--phase alpha-audit` | 6 (저자 3×시드 2) | `legacy-alpha-diagnostic`, evidence 파이프라인 |

7B 단일 모델 기준 샤드 폭이 GPU 수보다 작으므로, 남는 GPU는 같은 큐에
**시드 복제 런·추가 모델(llama31_8b 프로비저닝 후)·다른 config의 unit**을 함께
적재해 채운다. `make_units.py`가 못 만드는 임의 명령도 JSONL 한 줄이면 된다:

```json
{"unit_id": "chanbal2-s2026", "cmd": ["python", "-u", "experiments/gate_1p5b/gate.py", "--seed", "2026", "..."], "gpus": 1, "max_attempts": 1}
```

## 실패 triage — 봉인 러너의 partial 디렉토리는 자동 재시도로 안 살아난다

calibration/audit 러너는 설계상 **부분 산출물이 남은 run 디렉토리를 절대
재사용하지 않는다** (forensics 보존, seal append-only). 따라서 유닛이 도중에
죽으면 자동 재시도(2번째 attempt)는 "partial or pre-existing directory"
메시지로 몇 초 만에 실패하고 failed로 떨어진다 — 이건 데이터 보호가 작동한
것이지 큐 버그가 아니다. 복구 절차:

```bash
python experiments/cluster/workqueue.py status --queue <Q>   # failed의 log 경로 확인
less <log>                                                   # 죽은 원인 파악 (OOM? 노드?)
mv runs/channel_matrix_7b/calibration/.../<부분런> \
   runs/forensics/<부분런>.$(date +%s)                        # 부분 산출물 보존·이동
python experiments/cluster/workqueue.py retry-failed --queue <Q>
```

`requeue-stale`은 **해당 host의 워커가 정말 죽었는지 확인한 뒤에만** 실행할 것
(status가 보여주는 host에 들어가 프로세스 확인). NFS 지연으로 하트비트만 늦은
살아있는 런을 requeue하면 같은 run 디렉토리에 이중 실행이 붙을 수 있다.
기본 임계 30분은 하트비트 주기(60초)의 30배라 정상 지연으로는 안 걸린다.

## 병렬 폭 감각

calibration 유닛 1개 = gate 런 14개(7 objective × 2 설정) 직렬 ≈ GPU 1장을
오래 점유. 7B 단일 모델의 calibration 웨이브는 유닛 2개뿐이므로 플릿 전체가
아니라 **GPU 2장짜리 웨이브**다 — 이때 남는 GPU에 fidelity, 1.5B 시드 복제,
다른 config 유닛을 같이 적재해 채우는 게 맞다. 플릿이 진짜로 넓게 도는 건
audit(모델×저자 3)과 alpha-audit(모델×저자×시드 6) + 복수 모델부터다.

## 주의 (기존 CLAUDE.md 규칙과의 접점)

- audit 계열 unit은 러너 자체의 dirty-worktree 가드를 그대로 통과해야 하므로,
  **enqueue 전에 커밋 상태를 정리**할 것. 워커가 뜬 뒤 레포를 고치면 이후
  unit부터 바뀐 코드로 돌게 된다 — 봉인 페이즈 중 `git pull` 금지.
- run-tag를 쓰는 커스텀 unit(`gate.py` 등)은 재시도 시 같은 태그로 Exit 1이
  나므로 `max_attempts: 1`로 넣는 게 안전하다 (위 예시처럼).
- 큐 디렉토리는 `runs/` 아래라 git이 추적하지 않는다. 캠페인 산출물 회수는
  기존 방식(채팅 전달 → 세션 커밋) 유지.

## 부록: enqueue_table12.sh — Table 1/2 잔여 웨이브 원샷 적재

`experiments/cluster/enqueue_table12.sh`는 레포 루트에서 실행하는 운영자용
래퍼다. 워커를 띄우지 않고, `git pull`도 하지 않으며, 큐 상태는
`make_units.py --enqueue` 외에는 건드리지 않는다. 재실행 안전: 같은 unit id는
append-only 큐가 거부하고 스크립트는 "already enqueued"로 알려준다.

```bash
bash experiments/cluster/enqueue_table12.sh              # status (기본)
bash experiments/cluster/enqueue_table12.sh audit-7b     # 7B audit + alpha → wave2
bash experiments/cluster/enqueue_table12.sh audit-14b    # 14B audit + alpha → wave1_14b
bash experiments/cluster/enqueue_table12.sh wmdp         # WMDP fidelity+calibration → wave_wmdp
bash experiments/cluster/enqueue_table12.sh llama        # Llama-8B fidelity+calibration → wave_llama
bash experiments/cluster/enqueue_table12.sh rwku-audit   # RWKU audit → wave_rwku
```

- `status`: wave2 / wave1_14b / wave_wmdp / wave_llama / wave3_alpha /
  wave4_alpha 큐별 `workqueue.py status --brief` 요약 + `fleet_status.py` 안내.
- `audit-*`: enqueue 전에 (1) 해당 config의 objective_freeze가
  `status: frozen`인지 grep으로 확인(run_campaign.py의 게이트와 동일 기준),
  (2) worktree가 clean한지 확인(audit 러너가 dirty tree를 거부). alpha freeze가
  frozen이면 alpha-audit, 아직 draft면 alpha-development를 같은 큐에 적재.
- `llama`: 모델 경로(`/group-volume/models/Llama-3.1-8B-Instruct`) 부재 시
  `provision_llama.sh` 안내 후 중단. `rwku-audit`: fidelity 인증서 JSON이
  `runs/channel_matrix_rwku7b/fidelity/`에 없으면
  `experiments/diag/fd_fidelity.py --dataset rwku` 안내 후 중단.
- 적재 후 노드 투입은 언제나 수동:
  `bash experiments/cluster/launch_node.sh <큐>` (노드당 워커 8개, GPU 0-7).
  make_units가 만든 unit의 `max_attempts`는 그대로 둘 것.

웨이브 → 큐 → 논문 테이블 매핑:

| 테이블 | 큐 | 내용 |
|---|---|---|
| Table 1 | `wave2` | 7B TOFU audit (+ alpha-development) |
| Table 1 | `wave4_alpha` | 7B alpha-audit (alpha freeze 커밋 후) |
| Table 2 (14B 행) | `wave1_14b` | 14B TOFU audit |
| Table 2 (RWKU 행) | `wave_rwku` | RWKU 7B audit |
| Table 2 (WMDP 행) | `wave_wmdp` | WMDP 7B fidelity+calibration |
| Table 2 (Llama 행) | `wave_llama` | Llama-3.1-8B fidelity+calibration |
