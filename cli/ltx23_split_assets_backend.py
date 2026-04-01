from __future__ import annotations

from cli.ltx23_notebook_reference_backend import (
    NotebookReferencedPromptEncoder,
    build_pipeline,
)


def build_notebook_referenced_pipeline(**kwargs):
    return build_pipeline(**kwargs)


__all__ = [
    "NotebookReferencedPromptEncoder",
    "build_notebook_referenced_pipeline",
]
