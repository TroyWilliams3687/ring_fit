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


# ------------
# Custom Modules

# ------------


@click.command()
@click.pass_context
def sample(ctx: click.Context):
    """
    Sample stuff

    # Usage

    $ project_name sample

    """

    config = ctx.obj["config"]

    console.print('Sample Stuff!')

