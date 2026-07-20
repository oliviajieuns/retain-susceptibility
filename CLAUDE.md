# CLAUDE.md

## 사용자 클러스터 환경 (사내 H100 클라우드) — 다시 묻지 말 것

이 레포의 GPU 실행은 사용자의 사내 클러스터에서 사용자가 직접 수행한다.
Claude Code 세션 컨테이너에는 GPU가 없다 (CPU 검증만 가능).

### 접속/노드
- 노드 예: `run259705-ufail`, `run259706-ufail3` — 세션마다 노드가 바뀔 수 있음.
- **홈 디렉토리는 노드 로컬** (노드 간 공유 안 됨). 영구 저장은 `/group-volume`뿐.
- **레포 위치(공식)**: `/group-volume/jieuns.shin/retain-susceptibility` — 홈에 클론하지 말 것.
- 매 세션 시작 루틴:
  `source /group-volume/jieuns.shin/venvs/exp/bin/activate && cd /group-volume/jieuns.shin/retain-susceptibility`
- 노드당 H100 80GB × 2 (`nvidia-smi` 확인, 드라이버 535.129.03) — **CUDA_VISIBLE_DEVICES는 0/1만 유효**.
  "GPU 6장 할당" = 여러 노드에 걸친 것; 한 노드에서 3+ 지정하면 "No CUDA GPUs available"로 즉사.
- NVIDIA 드라이버 CUDA 12.2 (12020).
  - torch는 **cu12x 빌드만** 동작. cu130(torch 2.13 PyPI 기본)은 실패.
  - 검증된 조합: torch 2.5.1+cu121, torch 2.7.1+cu126 (둘 다 cuda=True 확인됨).

### venv — 공식은 하나뿐, 재설치 안내 금지
- **유일한 공식 venv**: `/group-volume/jieuns.shin/venvs/exp` (공유볼륨, 노드 무관).
  torch 2.5.1+cu121 동작 확인. transformers/datasets/pyyaml/pytest/sentence-transformers 설치됨.
- `~/.venv`는 폐기 — 언급하지 말 것.
- 사용법은 항상 이 한 줄: `source /group-volume/jieuns.shin/venvs/exp/bin/activate`
- 환경변수: `HF_HOME=/group-volume/data/hf_home`만 필요.
  **`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`은 설정하지 말 것** — 클러스터에서 HF Hub 접속 가능함이
  확인됨(2026-07-20, full/forget10_perturbed 자동 다운로드 성공). exp venv의 activate에 이 플래그를
  추가해뒀다면 제거할 것.

### 멀티노드 실행 규칙 (2026-07-20 사고 후 확정)
- 레포/runs 디렉토리는 전 노드 공유 → **로그는 반드시 `> <이름>_$(hostname).out`** (덮어쓰기 사고 방지).
- **run-tag는 시도마다 새로** — seal은 append-only라 같은 태그 재실행은 즉시 Exit 1.
- **띄우기 전 `nvidia-smi`로 대상 GPU가 빈 것 확인** — fp32 1.5B 게이트 런은 GPU당 1개(~58GB).
  겹배치 시 OOM으로 먼저 돌던 런이 죽는다 (03:02 diag 사망 사고).

### 네트워크 (사내망 차단 사항)
- GitHub SSH(포트 22) 차단 → HTTPS + PAT로 클론.
- Hugging Face Hub: **접속 가능** (다운로드 확인됨). HF_HOME만 공유볼륨으로 잡아 캐시 재사용.
- pip 인덱스는 사내 artifactory(`bart.sec.samsung.net`) 미러 — 일반 PyPI 패키지 설치 가능.
  `download.pytorch.org` 사용 불가, PyPI 기본 인덱스의 torch를 쓸 것.

### 모델 — 로컬 경로 사용 (허브 호출 금지)
- 위치: `/group-volume/models/` — `Qwen2.5-1.5B-Instruct`(2026-07-20 다운로드 완료 확인:
  config/safetensors/tokenizer 전부 있음), `Qwen2.5-7B-Instruct`, `Qwen3.6-27B`.
- 실행 시 항상 `--model /group-volume/models/<이름>` (로컬 경로 → 허브 HEAD 요청 원천 차단).
- 허브 직접 호출은 사내망에서 간헐적 connection reset → 장기 런 중 재시도 낭비. `HF_HUB_OFFLINE=1`을
  기본으로 걸고, 새 자산 다운로드 때만 잠시 해제.

### HF 데이터 캐시 (오프라인)
- 마운트: `/group-volume` (= `/home/user/group-volume`).
- 주 캐시: `HF_HOME=/group-volume/data/hf_home` — TOFU `full`(4000), `forget10_perturbed`(400) 포함
  전 컨피그 로드 확인 완료(2026-07-20). 사용자 확인: Qwen 모델도 준비돼 있음.
- 보조 캐시: `/group-volume/data/TOFU/hf_home` (다른 실험용, 이 레포는 주 캐시만 쓰면 됨).
- 데이터/모델 반입 작업 불필요 — 사용자에게 재확인시키지 말 것.

### Claude Code 세션 컨테이너 쪽 (참고)
- GPU 없음, HF/download.pytorch.org 프록시 차단, PyPI는 허용.
- CPU 테스트는 가능: `python -m pytest` → 69 passed, 2 skipped 기준.
