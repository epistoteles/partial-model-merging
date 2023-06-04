import rich
from rich.table import Table
from rich.panel import Panel
from rich.pretty import Pretty
from rich.console import Group
from rich.text import Text
from rich.align import Align


def get_header_panel(hyperparam_dict):
    table = Table(show_edge=False, show_lines=False, header_style="italic", box=rich.box.SIMPLE)

    for key in hyperparam_dict.keys():
        table.add_column(str(key), justify="left")

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
