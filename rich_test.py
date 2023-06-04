import time

import rich.box
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.table import Table
from rich.console import Group
from rich.pretty import Pretty
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)


class MyProgress(Progress):
    def get_renderables(self):
        table = Table(show_edge=False, show_lines=False, header_style="italic", box=rich.box.SIMPLE)

        table.add_column("dataset", justify="left")
        table.add_column("model", justify="left")
        table.add_column("size", justify="left")
        table.add_column("batch norm", justify="left")
        table.add_column("width", justify="left")
        table.add_column("epochs", justify="left")
        table.add_column("lr", justify="left")
        table.add_column("variant", justify="left")
        table.add_column("wandb", justify="left")
        table.add_column("test", justify="left")
        table.add_column("checkpoint_midway", justify="left")

        table.add_row(*[Pretty(x) for x in ["CIFAR10", "VGG", 11, False, 1, 100, 0.08, "a", True, True, False]])

        yield Panel(
            Group(
                Text("Training\n", style="bold orange1", justify="center"),
                # Text("CIFAR10 | VGG11 | batch_norm=True | 100 epochs | 0.08 lr", justify='center'),
                Align(table, align="center"),
            ),
            padding=1,
            width=170,
        )
        yield Panel(self.make_tasks_table(self.tasks), width=170)


progress = MyProgress(
    SpinnerColumn(speed=0.5),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=120),
    TaskProgressColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)

total = progress.add_task("Epochs", total=100)
train = progress.add_task("├─Training", total=50, start=False)
test = progress.add_task("└─Testing", total=10, start=False)

with progress:
    for epoch in range(100):
        progress.start_task(train)
        for i in range(50):
            progress.update(train, advance=1)
            time.sleep(0.1)
        progress.stop_task(train)

        progress.start_task(test)
        for i in range(10):
            progress.update(test, advance=1)
            time.sleep(0.1)
        progress.stop_task(test)

        progress.update(total, advance=1)
        progress.reset(train, start=False)
        progress.reset(test, start=False)
