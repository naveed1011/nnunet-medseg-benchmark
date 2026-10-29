"""Repo hygiene tests: these are the ones that catch a broken push.

They check the things a reader hits in the first 30 seconds: notebooks that do
not parse, links to files that do not exist, leftover `<YOUR-GITHUB-USERNAME>`
placeholders, syntax errors in any script.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _python_files():
    skip = {".git", "__pycache__", ".venv", "venv", "nnUNet_raw",
            "nnUNet_preprocessed", "nnUNet_results", "demo", "results"}
    for p in ROOT.rglob("*.py"):
        if any(part in skip for part in p.parts):
            continue
        yield p


def test_no_python_syntax_errors():
    failures = []
    for path in _python_files():
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    assert not failures, "\n".join(failures)


def test_notebooks_are_valid_and_placeholder_free():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks, "no notebooks found"
    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb.get("nbformat") == 4
        assert nb.get("cells"), f"{nb_path.name} has no cells"
        assert nb["metadata"]["kernelspec"]["name"] == "python3"
        text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        assert "YOUR-GITHUB-USERNAME" not in text, f"{nb_path.name} still has a repo placeholder"
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                # every code cell must at least be valid Python once IPython
                # magics and shell escapes are stripped (joining shell line
                # continuations first, since `!cmd \` spans two lines)
                raw = "".join(cell["source"]).replace("\\\n", " ")
                src = "\n".join(
                    line for line in raw.splitlines()
                    if not line.lstrip().startswith(("!", "%", "#"))
                )
                if src.strip():
                    compile(src, f"{nb_path.name}:cell", "exec")


def test_readme_relative_links_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = []
    for target in re.findall(r"\]\(([^)\s]+)\)", readme):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        clean = target.split("#")[0]
        if not clean:
            continue
        if not (ROOT / clean).exists():
            missing.append(target)
    assert not missing, f"README links to missing files: {missing}"


def test_requirements_are_parseable():
    lines = [ln.strip() for ln in (ROOT / "requirements.txt").read_text().splitlines()]
    deps = [ln for ln in lines if ln and not ln.startswith("#")]
    assert any(d.startswith("nnunetv2") for d in deps)
    assert any(d.startswith("nibabel") for d in deps)


def test_gitignore_excludes_data_and_keeps_figures():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("nnUNet_raw/", "nnUNet_preprocessed/", "nnUNet_results/", "data/raw/"):
        assert pattern in text, f"{pattern} missing from .gitignore"
    # a bare `figures/` rule also matches docs/figures/ and silently drops the
    # very images the README links to — the repo must not have one
    assert "\nfigures/\n" not in text
    assert "!docs/figures/" in text, "docs/figures must be explicitly kept"
    assert "!docs/results/" in text, "docs/results must be explicitly kept"


def test_core_modules_import_without_heavy_deps():
    """src.metrics / src.compare must import with the light dependency set."""
    import importlib
    for mod in ("src.metrics", "src.compare", "src.figures"):
        importlib.import_module(mod)


def test_run_pipeline_cli_is_wired():
    import subprocess
    proc = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    for cmd in ("doctor", "demo", "plan", "train", "predict", "evaluate", "compare"):
        assert cmd in proc.stdout


def test_docs_have_no_unfilled_placeholders():
    docs = list((ROOT / "docs").glob("*.md"))
    assert docs
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        assert "<YOUR-GITHUB-USERNAME>" not in text, f"{doc.name} has a placeholder"
