"""PlasMEEP: a plasma-themed wrapper for Meep.

Exposes the primary :class:`Plasmeep` class at the package level so it can be
imported after ``pip install -e .`` regardless of the current working
directory.
"""

from plasmeep.lib import Plasmeep

__all__ = ["Plasmeep"]
