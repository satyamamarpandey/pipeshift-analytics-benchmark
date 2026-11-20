"""Redact source-data values that library error messages quote (public-release hygiene).

Some generated programs raise exceptions whose message embeds a value taken from the input records, for example
pandas' ``time data "02/01/2025 02:32:01 AM" doesn't match format`` or ``Could not convert string '9.318.4...'``
(a long run of concatenated fare values). The experiment scores the exception only as an abstention and never reads
the message, so the quoted value is replaced by ``<value>`` in the released records. Counts, column names and
generic messages are kept.
"""
from __future__ import annotations

import re

_RULES = [
    # pandas / numpy: could not convert string 'xxx' to float / Could not convert string 'xxx' to numeric
    (re.compile(r"((?:could not|Could not) convert string(?: to float:)? )'[^']*(?:'|$)"), r"\1'<value>'"),
    # pandas datetime parsing: time data "xxx" doesn't match format ...
    (re.compile(r'(time data )"[^"]*(?:"|$)'), r'\1"<value>"'),
    (re.compile(r"(time data )'[^']*(?:'|$)"), r"\1'<value>'"),
    # pandas: Unknown datetime string format, unable to parse: xxx
    (re.compile(r'(unable to parse(?: string)?:? )["\']?[^"\',)]*'), r'\1<value>'),
    # pandas: Given date string "xxx" not likely a datetime
    (re.compile(r'(date string )"[^"]*"'), r'\1"<value>"'),
]


def redact_values(text: str) -> str:
    for pat, repl in _RULES:
        text = pat.sub(repl, text)
    return text
