"""Test package.

Declared so shared fixture helpers resolve under a single module name for both pytest
and mypy, rather than being discovered twice as ``fixtures`` and ``tests.fixtures``.
"""
