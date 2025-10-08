"""Explains Codebase from Flattened Markdown"""
import sys


def __main__():  # pragma: no cover
    if len(sys.argv) != 2:
        print("Usage: python -m flatr.explain <github_repo_url>")
        sys.exit(1)

    # TODO: flatten the repo and explain it with LLM
