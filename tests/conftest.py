from types import SimpleNamespace

import pytest


def pytest_addoption(parser):
    parser.addoption("--sampling-device", default="cpu")


@pytest.fixture
def sampling_device(request):
    return request.config.getoption("--sampling-device")


@pytest.fixture
def torch():
    return pytest.importorskip(
        "torch", reason="Run tensor tests in the remote test Image."
    )


@pytest.fixture
def inputs(torch):
    return {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }


@pytest.fixture
def model_factory(torch):
    class ScriptedModel:
        def __init__(self, tokens, context=16, nonfinite=False):
            self.tokens = tokens
            self.device = torch.device("cpu")
            self.config = SimpleNamespace(vocab_size=8, max_position_embeddings=context)
            self.calls = []
            self.nonfinite = nonfinite

        def __call__(self, *, input_ids, attention_mask, use_cache):
            assert use_cache is False
            assert not torch.is_grad_enabled()
            assert attention_mask.shape == input_ids.shape
            assert attention_mask.eq(1).all().item()
            token = self.tokens[len(self.calls)]
            self.calls.append(input_ids[0].tolist())
            logits = torch.zeros(1, input_ids.shape[1], self.config.vocab_size)
            logits[0, -1, token] = float("nan") if self.nonfinite else 10
            return SimpleNamespace(logits=logits)

    return ScriptedModel
