#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logging utilities for debug-level output.

Usage:
    set_debug(level)  -- set global verbosity (higher = more output)
    log(level, *args) -- print if debug > level
    is_debug(level)   -- returns True if debug > level
"""

import re
import string
import sys

# logging level
debug = 4


def log(level: int, *args):
    global debug
    if debug > level:
        print(" " * (level - 1), *args)


def is_debug(level):
    global debug
    return debug > level


def set_debug(level):
    global debug
    debug = level
