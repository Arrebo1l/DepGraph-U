# Contributing

Bug reports and focused reproduction improvements are welcome. State the repository commit, upstream revision, environment, hardware, exact command, and failure. Exclude credentials and private data from logs.

Before submitting changes, run:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/verify_release.py
```

Keep historical evidence immutable. A corrected result belongs in a new record with its source, denominator, evaluation rule, and an erratum. Do not silently replace an unsuccessful run or change a correctness gate to improve a speedup.

For a new GPU adapter, report the fixed reference, synchronized timing scope, warmups, RNG control, output checks, and both positive and negative outcomes. Matching aggregate scores alone is insufficient evidence of numerical equivalence.
