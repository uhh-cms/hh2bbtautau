# coding: utf-8

"""
Custom, common parameters.
"""

import luigi


table_format_param = luigi.Parameter(
    default="fancy_grid",
    description="a tabulate table format; default: 'fancy_grid'",
)
escape_markdown_param = luigi.BoolParameter(
    default=False,
    description="escape some characters for markdown; default: False",
)
