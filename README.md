# Polymarket Market-Making Agent Harness

An open-source research harness for stress-testing long-running autonomous coding
agents on a real trading problem: market making in Polymarket 15-minute binary
options. The project couples a realistic backtest model with a growing library
of market-making strategies, then lets autonomous agents iterate end-to-end.

## Contributions

- Proper backtesting model for Polymarket-style binary markets
- Market-making strategies (baseline, symmetric spread, inventory-aware,
  complement-aware, adaptive spread, lifecycle-aware, order-flow-aware)

## Coming Soon

- Lightweight data recorder with 100% coverage for order flow and order book at > 1Hz

## Repository Layout

- `mm/` - backtest engine, fill models, evaluation tools, and strategy library
- `scripts/ralph/` - long-running agent loop (Ralph), prompts, and PRD backlog

## Getting Started

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Running the Autonomous Loop (Ralph)

Ralph drives iterative, long-running coding sessions from a local PRD.

1) Make sure the Claude Code CLI is installed and authenticated (`claude` on PATH).
2) Run the loop:

```bash
./scripts/ralph/ralph.sh --tool claude 10
```

Ralph reads:
- `scripts/ralph/prd.json` for the prioritized story list
- `scripts/ralph/CLAUDE.md` for the agent instructions
- `scripts/ralph/progress.txt` for cumulative progress logs

## Backtesting

The backtest engine and fill models live under `mm/src/mm/backtest_engine`, with
strategy implementations in `mm/src/mm/strategies`. Evaluation helpers live in
`mm/src/mm/eval`.

## Contributing

Issues and PRs are welcome. If you are proposing a new strategy or data pipeline,
please open an issue with a short design note before submitting a large change.

## License

TBD.
