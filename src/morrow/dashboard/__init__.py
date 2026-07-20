"""Localhost results dashboard."""

from .app import render_page
from .server import runtime_info, serve

__all__ = ["serve", "render_page", "runtime_info"]
