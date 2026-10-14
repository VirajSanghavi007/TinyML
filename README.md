# TinyML

## Branches

- `prod`: production-ready code
- `dev`: integration branch for ongoing work
- `test`: testing branch

## Layout

- `test/`: test suite (run with `uv run --with pytest pytest test`)
- `.github/workflows/ci.yml`: CI pipeline, runs on pushes and PRs to `prod`, `dev`, `test`
