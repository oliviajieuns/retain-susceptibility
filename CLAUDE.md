# CLAUDE.md

## 사용자 클러스터 환경 (사내 H100 클라우드) — 다시 묻지 말 것

이 레포의 GPU 실행은 사용자의 사내 클러스터에서 사용자가 직접 수행한다.
Claude Code 세션 컨테이너에는 GPU가 없다 (CPU 검증만 가능).

### 접속/노드
- 노드 예: `run259705-ufail`, `run259706-ufail3` — 세션마다 노드가 바뀔 수 있음.
- 노드당 H100 80GB × 2, NVIDIA 드라이버 CUDA 12.2 (12020).
  - torch는 **cu12x 빌드만** 동작. cu130(torch 2.13 PyPI 기본)은 실패.
  - 검증된 조합: torch 2.5.1+cu121, torch 2.7.1+cu126 (둘 다 cuda=True 확인됨).

### venv — 이미 설치 완료, 재설치 안내 금지
- 영구(공유볼륨): `/group-volume/jieuns.shin/venvs/exp` — torch 2.5.1+cu121 동작 확인.
- 홈(노드 공유 여부 확인됨, run259706): `~/.venv` — torch 2.7.1+cu126 동작 확인.
- activate만 하면 됨: `source /group-volume/jieuns.shin/venvs/exp/bin/activate`
- 환경변수는 activate 스크립트 끝에 추가돼 있음(또는 추가 예정):
  `HF_HOME`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`

### 네트워크 (사내망 차단 사항)
- GitHub SSH(포트 22) 차단 → HTTPS + PAT로 클론.
- Hugging Face 차단 → 오프라인 캐시 사용 (아래).
- pip 인덱스는 사내 artifactory(`bart.sec.samsung.net`) 미러 — 일반 PyPI 패키지 설치 가능.
  `download.pytorch.org` 사용 불가, PyPI 기본 인덱스의 torch를 쓸 것.

### HF 데이터 캐시 (오프라인)
- 마운트: `/group-volume` (= `/home/user/group-volume`).
- 주 캐시: `HF_HOME=/group-volume/data/hf_home` — TOFU forget10/retain90 확인됨.
- 보조 캐시: `/group-volume/data/TOFU/hf_home` — forget10_perturbed, real_authors 등 더 많은 컨피그.
- 미해결(2026-07-20): TOFU `full` 컨피그와 Qwen2.5-1.5B-Instruct 모델이 캐시에 있는지 미확인.
  없으면 외부망에서 받아 volume 반입 필요. `full`을 retain90+forget10 이어붙이기로 대체하지 말 것
  (코드가 원본 행 순서 200저자×20행에 의존).

### Claude Code 세션 컨테이너 쪽 (참고)
- GPU 없음, HF/download.pytorch.org 프록시 차단, PyPI는 허용.
- CPU 테스트는 가능: `python -m pytest` → 69 passed, 2 skipped 기준.
