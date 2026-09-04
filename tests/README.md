# Tests

I will add tests as reusable behavior emerges. The current scaffold has no tests.

Useful future checks include sampling edge cases, cached versus uncached logits,
batched versus individual outputs, and scheduler admission/completion behavior.
Ordinary unit tests will remain local and free of cloud charges. Remote GPU
experiments will run explicitly and be documented separately.

Once tests exist, the local test command will be `python -m pytest` from the
repository root.
