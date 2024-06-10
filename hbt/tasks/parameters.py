# coding: utf-8

"""
Custom, common parameters.
"""

import getpass

import luigi


table_format_param = luigi.Parameter(
    default="fancy_grid",
    description="a tabulate table format; default: 'fancy_grid'",
)
escape_markdown_param = luigi.BoolParameter(
    default=False,
    description="escape some characters for markdown; default: False",
)
user_parameter_inst = luigi.Parameter(
    default=getpass.getuser(),
    description="the user running the current task, mainly for central schedulers to distinguish "
    "between tasks that should or should not be run in parallel by multiple users; "
    "default: current user",
)
