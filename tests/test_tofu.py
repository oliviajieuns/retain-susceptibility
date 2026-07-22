"""TOFU adapter checks against the locally cached dataset. Skipped when the
dataset (or datasets lib) is unavailable."""
import pytest

datasets = pytest.importorskip("datasets")

from rsus.data.tofu import (  # noqa: E402
    FORGET10_FIRST_AUTHOR,
    QA_PER_AUTHOR,
    format_qa,
    load_tofu_rows,
    tofu_request,
)
from rsus.data.base import Example  # noqa: E402
from rsus.losses import IGNORE  # noqa: E402


class MockTokenizer:
    eos_token_id = 9

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(c) % 50) for c in text[:40]]}


@pytest.fixture(scope="module")
def rows():
    try:
        return load_tofu_rows()
    except Exception as e:  # dataset not cached / offline
        pytest.skip(f"TOFU unavailable: {e}")


def test_format_qa_masks_prompt():
    ids, labels = format_qa("Who?", "Someone.", MockTokenizer())
    assert (labels == IGNORE).sum() > 0
    n_prompt = int((labels == IGNORE).sum())
    assert labels[n_prompt:].tolist() == ids[n_prompt:].tolist()
    assert labels[-1].tolist() == MockTokenizer.eos_token_id


def test_paraphrase_mapping():
    from rsus.data.tofu import load_tofu_paraphrases

    try:
        paras = load_tofu_paraphrases(MockTokenizer())
    except Exception as e:
        pytest.skip(f"forget10_perturbed unavailable: {e}")
    assert len(paras) == 400
    assert "tofu-3600" in {k.rsplit("-", 0)[0] for k in paras} or "tofu-3600" in paras
    ex = paras["tofu-3600"]
    assert ex.group == "author-180"
    assert ex.example_id.endswith("-para")
    assert (ex.labels == IGNORE).sum() > 0


def test_request_construction(rows):
    tok = MockTokenizer()
    # build examples for a small slice only (speed): author 180 + 3 retained
    idxs = list(range(0, 3 * QA_PER_AUTHOR)) + list(
        range(FORGET10_FIRST_AUTHOR * QA_PER_AUTHOR, (FORGET10_FIRST_AUTHOR + 1) * QA_PER_AUTHOR)
    )
    examples = []
    for idx in idxs:
        ids, labels = format_qa(rows[idx]["question"], rows[idx]["answer"], tok)
        examples.append(
            Example(f"tofu-{idx:04d}", ids, labels, group=f"author-{idx // QA_PER_AUTHOR:03d}")
        )
    req = tofu_request(FORGET10_FIRST_AUTHOR, examples)
    assert len(req.forget) == QA_PER_AUTHOR
    assert len(req.universe) == 3 * QA_PER_AUTHOR
    assert all(e.group != f"author-{FORGET10_FIRST_AUTHOR:03d}" for e in req.universe.examples)
    capped = tofu_request(FORGET10_FIRST_AUTHOR, examples, universe_authors=2, seed=0)
    assert len(capped.universe) == 2 * QA_PER_AUTHOR
    fixed = tofu_request(
        FORGET10_FIRST_AUTHOR,
        examples,
        universe_authors=2,
        candidate_authors=[0, 2],
    )
    assert {example.group for example in fixed.universe.examples} == {
        "author-000", "author-002"
    }
