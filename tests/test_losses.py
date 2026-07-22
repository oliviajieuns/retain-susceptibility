import torch
import torch.nn.functional as F

from rsus.data.base import collate
from rsus.losses import IGNORE, seq_mean_answer_nll, seq_sum_answer_nll, token_answer_nll


def test_seq_mean_matches_manual(tiny_model, req):
    batch = collate(list(req.universe.examples[:3]))
    losses = seq_mean_answer_nll(tiny_model, batch)
    with torch.no_grad():
        logits = tiny_model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
    for i in range(3):
        lg, tg = logits[i, :-1], batch["labels"][i, 1:]
        keep = tg != IGNORE
        manual = F.cross_entropy(lg[keep], tg[keep])
        assert torch.allclose(losses[i], manual, atol=1e-10), i


def test_seq_sum_is_mean_times_answer_length(tiny_model, req):
    examples = list(req.universe.examples[:3])
    batch = collate(examples)
    mean = seq_mean_answer_nll(tiny_model, batch)
    summed = seq_sum_answer_nll(tiny_model, batch)
    counts = torch.tensor([example.n_answer_tokens() for example in examples], dtype=mean.dtype)
    assert torch.allclose(summed, mean * counts, atol=1e-10)


def test_token_index_map_batch_invariant(tiny_model, req):
    exs = list(req.universe.examples[:4])
    b_all = collate(exs)
    with torch.no_grad():
        flat_all, idx_all = token_answer_nll(tiny_model, b_all)
    by_key_all = dict(zip(idx_all, flat_all.tolist()))
    # Same examples, shuffled composition across two batches
    with torch.no_grad():
        f1, i1 = token_answer_nll(tiny_model, collate([exs[2], exs[0]]))
        f2, i2 = token_answer_nll(tiny_model, collate([exs[3], exs[1]]))
    by_key_shuffled = dict(zip(i1 + i2, torch.cat([f1, f2]).tolist()))
    assert by_key_all.keys() == by_key_shuffled.keys()
    for k in by_key_all:
        assert abs(by_key_all[k] - by_key_shuffled[k]) < 1e-10, k
    # index map counts answer tokens per example
    n_ans = sum(e.n_answer_tokens() for e in exs)
    assert len(idx_all) == n_ans == flat_all.numel()
