"""Sealed audit-fold scores with an append-only ledger.

Audit-fold susceptibility scores are computed once, written under seals/, and
recorded as 'sealed' in the ledger. They may be read only after an 'opened'
ledger entry, and unsealing requires the DONE markers of every pre-fixed
third-party trajectory — honest-by-construction ordering, auditable in the
released artifact.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class SealedError(RuntimeError):
    pass


def _ledger_entries(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    with open(ledger, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append(ledger: Path, entry: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _seal_path(seals_dir: Path, request_id: str, scorer: str) -> Path:
    return Path(seals_dir) / request_id / f"{scorer}.json"


def seal_scores(
    seals_dir: str | Path,
    ledger: str | Path,
    request_id: str,
    scorer: str,
    scores: dict[str, float],
) -> str:
    """Write audit-fold scores and record them as sealed. Returns the sha."""
    path = _seal_path(Path(seals_dir), request_id, scorer)
    if path.exists():
        raise SealedError(f"seal already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(scores, sort_keys=True)
    sha = hashlib.sha256(body.encode()).hexdigest()
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    _append(
        Path(ledger),
        {
            "status": "sealed",
            "request": request_id,
            "scorer": scorer,
            "sha": sha,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return sha


def unseal(
    seals_dir: str | Path,
    ledger: str | Path,
    request_id: str,
    scorer: str,
    done_markers: list[str | Path],
) -> dict[str, float]:
    """Open a seal after verifying every trajectory DONE marker exists."""
    missing = [str(m) for m in done_markers if not Path(m).exists()]
    if missing:
        raise SealedError(f"cannot unseal; missing DONE markers: {missing}")
    entries = _ledger_entries(Path(ledger))
    if not any(
        e["status"] == "sealed" and e["request"] == request_id and e["scorer"] == scorer
        for e in entries
    ):
        raise SealedError(f"no sealed entry for {request_id}/{scorer}")
    _append(
        Path(ledger),
        {
            "status": "opened",
            "request": request_id,
            "scorer": scorer,
            "markers": sorted(str(m) for m in done_markers),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return read_scores(seals_dir, ledger, request_id, scorer)


def read_scores(
    seals_dir: str | Path, ledger: str | Path, request_id: str, scorer: str
) -> dict[str, float]:
    """Read previously sealed scores; requires an 'opened' ledger entry."""
    entries = _ledger_entries(Path(ledger))
    if not any(
        e["status"] == "opened" and e["request"] == request_id and e["scorer"] == scorer
        for e in entries
    ):
        raise SealedError(f"{request_id}/{scorer} is sealed (no 'opened' ledger entry)")
    path = _seal_path(Path(seals_dir), request_id, scorer)
    with open(path, "r", encoding="utf-8") as f:
        body = f.read()
    sha = hashlib.sha256(body.encode()).hexdigest()
    sealed = [
        e
        for e in entries
        if e["status"] == "sealed" and e["request"] == request_id and e["scorer"] == scorer
    ]
    if sealed and sealed[-1]["sha"] != sha:
        raise SealedError(f"seal content hash mismatch for {request_id}/{scorer}")
    return json.loads(body)
