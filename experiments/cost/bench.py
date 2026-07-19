"""Table-4 profiling-cost bench: run each implementation repeatedly on the
same frozen pool and report median [IQR] wall time, peak memory, and
candidate-token throughput from the scorers' own CostRecords.

GPU run (7B/14B) lands at N5; --smoke validates the harness on CPU.

  python experiments/cost/bench.py --smoke
  python experiments/cost/bench.py --model Qwen/Qwen2.5-7B-Instruct \
      --device cuda --universe-authors 30 --repeats 5
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.blocks import mlp_down_last_layers  # noqa: E402
from rsus.probe.base import ProbeSpec, get_scorer  # noqa: E402

IMPLEMENTATIONS = ["fd", "jvp", "vmap_graddot", "streaming_backward"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--universe-authors", type=int, default=30)
    p.add_argument("--author", type=int, default=180)
    p.add_argument("--block-last-n", type=int, default=8)
    p.add_argument("--eta", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def build_world(a):
    if a.smoke:
        sys.path.insert(0, str(ROOT / "tests"))
        from conftest import build_tiny, make_example

        from rsus.data.base import CandidateUniverse, Request

        model = build_tiny(0)
        gen = torch.Generator().manual_seed(7)
        forget = [make_example(gen, f"f{i:02d}") for i in range(4)]
        cands = [make_example(gen, f"c{i:02d}") for i in range(16)]
        req = Request.build("bench-smoke", forget, CandidateUniverse.freeze(cands))
        block = mlp_down_last_layers(model, 1)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from rsus.data.tofu import load_tofu_examples, tofu_request

        tokenizer = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
        model = model.to(a.device).eval()
        examples = load_tofu_examples(tokenizer)
        req = tofu_request(a.author, examples, universe_authors=a.universe_authors, seed=a.seed)
        block = mlp_down_last_layers(model, a.block_last_n)
    return model, req, block


def med_iqr(xs):
    xs = sorted(xs)
    med = statistics.median(xs)
    q1 = xs[len(xs) // 4]
    q3 = xs[(3 * len(xs)) // 4]
    return med, q1, q3


def main():
    a = parse_args()
    model, req, block = build_world(a)
    spec = ProbeSpec(block=block, eta=a.eta, batch_size=a.batch_size)
    out = ROOT / "runs" / ("bench_smoke" if a.smoke else f"bench_{a.model.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in IMPLEMENTATIONS:
        walls, mems, thrpts = [], [], []
        for r in range(a.repeats):
            rec = get_scorer(name)(model, req, spec).cost
            walls.append(rec.wall_s)
            mems.append(rec.peak_mem_bytes)
            thrpts.append(rec.tokens_fwd / rec.wall_s if rec.wall_s > 0 else 0.0)
        w = med_iqr(walls)
        t = med_iqr(thrpts)
        rows.append(
            {
                "impl": name,
                "wall_s_med": round(w[0], 4), "wall_s_q1": round(w[1], 4), "wall_s_q3": round(w[2], 4),
                "peak_mem_mb": round(max(mems) / 1e6, 1),
                "tok_per_s_med": round(t[0], 1),
                "cand_reverse_mode": name in ("vmap_graddot", "streaming_backward"),
            }
        )
        print(rows[-1])
    with open(out / "table4.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'table4.csv'}")


if __name__ == "__main__":
    main()
