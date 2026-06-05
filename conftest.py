"""
conftest.py — pytest bootstrap for this project.

This project is run as a *script directory*, not an installed package: every
module uses flat absolute imports (``from graph import ...``) and relies on the
project folder being on ``sys.path``. ``app.py`` / ``main.py`` arrange this with
``sys.path.insert(0, ...)`` at runtime.

pytest doesn't run those entrypoints, so we replicate that one line here. pytest
auto-imports this file before collecting any tests, which puts the project folder
on ``sys.path`` and lets ``pytest test_agent.py`` resolve ``graph``, ``tools``,
``state``, etc. — matching the command documented in CLAUDE.md.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
