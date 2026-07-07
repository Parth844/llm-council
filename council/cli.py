"""CLI: python -m council "question" --rounds 2 [--json]"""

from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from council.client import NIMClient
from council.config import DEFAULT_CONFIG_PATH, load_api_key, load_config
from council.engine import DebateEngine, DebateEvent

ALIAS_COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m council", description="Run an LLM Council debate."
    )
    parser.add_argument("question", help="The question to debate")
    parser.add_argument("--rounds", type=int, default=2, help="Total rounds (default 2)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to models.yaml")
    parser.add_argument("--json", action="store_true", help="Emit the full result as JSON")
    parser.add_argument(
        "--no-health-check", action="store_true",
        help="Skip startup model probes (saves ~1 request per model)",
    )
    return parser


async def run(args: argparse.Namespace) -> int:
    console = Console(stderr=args.json)  # keep stdout clean for --json
    config = load_config(args.config)
    client = NIMClient(
        api_key=load_api_key(),
        base_url=config.base_url,
        requests_per_minute=config.requests_per_minute,
    )
    colors = {
        m.alias: ALIAS_COLORS[i % len(ALIAS_COLORS)]
        for i, m in enumerate(config.council)
    }

    def on_event(ev: DebateEvent) -> None:
        if ev.type == "round_started":
            label = "Independent answers" if ev.round == 1 else "Cross-examination"
            console.print(Rule(f"[bold]Round {ev.round} — {label}[/bold]"))
        elif ev.type in ("model_answered", "critique"):
            color = colors.get(ev.alias or "", "white")
            console.print(
                Panel(
                    ev.text or "",
                    title=f"[{color}]{ev.alias}[/{color}]  ({ev.latency_s:.1f}s)",
                    border_style=color,
                )
            )
        elif ev.type == "model_failed":
            console.print(f"[red]✗ {ev.alias or ev.model_id} failed:[/red] {ev.text}")
        elif ev.type == "synthesis":
            console.print(Rule("[bold]Chief Justice — Verdict[/bold]"))
            console.print(Panel(ev.text or "", border_style="bold white"))
        elif ev.type == "error":
            console.print(f"[bold red]ERROR:[/bold red] {ev.text}")

    engine = DebateEngine(
        config, client, on_event=on_event, skip_health_check=args.no_health_check
    )
    try:
        result = await engine.run(args.question, rounds=args.rounds)
    finally:
        await client.aclose()

    if args.json:
        print(result.model_dump_json(indent=2))
    return 0 if result.synthesis is not None else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
