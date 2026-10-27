# bench-atlas

Tools for analysing and comparing continuous black-box optimization benchmarks:
ELA, LON, diversity, cost and algorithm discrimination.

> Early development — the CLI subcommands are placeholders for now.

## Installation

```bash
uv sync
```

## Usage

```bash
uv run bench-atlas --help
```

Planned subcommands: `ela`, `lon`, `timing`, `run`, `metrics`, `report`.

## Problems

All benchmark problems are [IOH](https://github.com/IOHprofiler/IOHexperimenter)
problems, analysed on [0, 1]^d (see `bench_atlas.problems`). Every place where we
deviate from plain IOH is listed in [docs/ioh-deviations.md](docs/ioh-deviations.md).

## Development

```bash
uv sync                       # install the package with dev dependencies
uv run pre-commit install     # enable git hooks
uv run pre-commit run --all-files
uv run pytest
```

## License

MIT
