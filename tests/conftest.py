"""Pytest config: skip publication matplotlib style so CI doesn't need LaTeX.

Set before any test module imports ldv_analysis.config. The science+ieee
style enables text.usetex which needs a system LaTeX install we don't
require for the test suite.
"""

import os

os.environ.setdefault("LDV_NO_STYLE", "1")
