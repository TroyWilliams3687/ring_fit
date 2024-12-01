#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
"""

# ------------
# System Modules - Included with Python

# ------------
# 3rd Party - From pip

import click

from rich.console import Console; console = Console()

# install the rich traceback to colorize exceptions. This should only be on the main entry point
from rich.traceback import install; install(show_locals=False)

# ------------
# Custom Modules

from .command_sample import sample


# -------------


@click.group()
@click.version_option()
@click.pass_context

def main(ctx: click.Context):
    """ """

    ctx.ensure_object(dict)


# Add the child menu options
main.add_command(sample)

