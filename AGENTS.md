# Repository conventions

## Documentation voice

- Use first person singular for project goals, progress, decisions, and reflections.
- Use neutral language for technical descriptions and reproduction instructions.
- Do not address the maintainer as "the user" or write instructions aimed at them.
- Describe incomplete work as planned project work rather than a task for the reader.

## Experiment results

- Keep measured values, output dumps, and result tables out of stage READMEs.
- Store canonical machine-readable measurements under `benchmarks/results/`.
- Make benchmark modes write their canonical result artifact automatically.
- Use stage READMEs for the plan, method, reproduction command, interpretation,
  and links to result artifacts.
- Keep large or disposable runs under `benchmarks/results/raw/`.
