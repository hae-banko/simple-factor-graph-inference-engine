"""Setup shim for editable installs.

``pyproject.toml`` handles the build config; this file exists only so that
``pip install -e .`` works (setuptools requires it for editable mode).
"""

from setuptools import setup

setup()
