"""
utils/progress.py
------------------
Thin wrapper around `rich` for consistent progress bars and a summary
banner, used by main.py to give the "beautiful progress" the spec asks for.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()


def banner(title: str, subtitle: str = "") -> None:
    text = f"[bold]{title}[/bold]"
    if subtitle:
        text += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel.fit(text, border_style="cyan"))


@contextmanager
def spinner(message: str) -> Iterator[Progress]:
    """A single indeterminate spinner for stages without a known length."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    with progress:
        progress.add_task(message, total=None)
        yield progress


@contextmanager
def bar(message: str, total: int) -> Iterator[Progress]:
    """A determinate progress bar for stages with a known number of steps."""
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    with progress:
        task_id = progress.add_task(message, total=total)
        yield progress, task_id  # type: ignore[misc]


def summary_table(rows: dict) -> None:
    """Print a final summary table (video info, timings, output path...)."""
    table = Table(title="Dubbing Summary", show_header=False, border_style="green")
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    for key, value in rows.items():
        table.add_row(str(key), str(value))
    console.print(table)
