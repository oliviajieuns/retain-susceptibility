"""CPU test fixtures: tiny double-precision causal LM + synthetic requests.

Double precision makes finite-difference truncation the only material error
term, so probe invariants can assert exact rank agreement.
"""
from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from rsus.blocks import mlp_down_last_layers
from rsus.data.base import CandidateUniverse, Example, Request
from rsus.losses import IGNORE
from rsus.probe.base import ProbeSpec

VOCAB = 128


def build_tiny(seed: int = 0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    return LlamaForCausalLM(cfg).double().eval()


@pytest.fixture(scope="session")
def tiny_model():
    return build_tiny(0)


def make_example(gen: torch.Generator, eid: str, seq_len: int = 16, prompt_len: int = 8) -> Example:
    ids = torch.randint(3, VOCAB, (seq_len,), generator=gen)
    labels = ids.clone()
    labels[:prompt_len] = IGNORE
    return Example(example_id=eid, input_ids=ids, labels=labels)


@pytest.fixture()
def req() -> Request:
    gen = torch.Generator().manual_seed(7)
    forget = [make_example(gen, f"f{i:02d}") for i in range(4)]
    cands = [make_example(gen, f"c{i:02d}") for i in range(12)]
    return Request.build("req-test", forget, CandidateUniverse.freeze(cands))


@pytest.fixture()
def spec(tiny_model) -> ProbeSpec:
    return ProbeSpec(block=mlp_down_last_layers(tiny_model, 1), eta=1e-4, batch_size=5)
