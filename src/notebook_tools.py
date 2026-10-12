"""
Turn a notebook source written as a plain Python file into an executed .ipynb.

Source format ("percent" cells): a line  # %% [markdown]  starts a markdown cell whose lines are written as
comments ("# text"); a line  # %%  starts a code cell. The .py file is the source of truth and diffs cleanly; the
.ipynb next to it holds the executed outputs and figures.

    uv run src/notebook_tools.py notebooks/01_eda/01_coverage.py [more.py ...]

Notebooks run with the project root as the working directory.
"""

import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(re.sub(r"^# ?", "", line) for line in body.split("\n"))
            cells.append(nbformat.v4.new_markdown_cell(body.strip()))
        elif body.strip():
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.split("\n"):
        m = re.match(r"^# %%\s*(\[markdown\])?\s*$", line)
        if m:
            flush()
            kind, buf = ("markdown" if m.group(1) else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def build(path, execute=True):
    path = Path(path)
    nb = nbformat.v4.new_notebook(cells=parse(path.read_text()))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if execute:
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(Path(__file__).resolve().parent.parent)}}).execute()
    out = path.with_suffix(".ipynb")
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("built", build(p))
