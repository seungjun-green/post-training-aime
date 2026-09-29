import ast
import base64
import io
from pathlib import Path
from zipfile import ZipFile

import nbformat
import pytest


@pytest.mark.parametrize(
    "name",
    [
        "translate_datasets",
        "evaluate_baseline",
        "translate_datasets_qwen",
        "translate_datasets_deepseek",
        "translate_datasets_openai",
        "translate_datasets_exaone",
        "prepare_english_datasets",
    ],
)
def test_standalone_notebook_syntax_and_bundle_matches_sources(name):
    path = Path("notebooks") / f"{name}.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    bundle = None
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        compile(cell.source, str(path), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        tree = ast.parse(cell.source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "BUNDLE" for t in node.targets
            ):
                bundle = ast.literal_eval(node.value)
        assert cell.execution_count is None and not cell.outputs
    assert bundle is not None
    with ZipFile(io.BytesIO(base64.b64decode(bundle))) as z:
        for member in z.namelist():
            # The baseline notebook is intentionally frozen. Its unused bundled
            # translation/data-preparation files need not track later data fixes.
            if name == "evaluate_baseline" and (
                member.startswith("pipeline/")
                or member in {"configs/translation.yaml", "configs/datasets.yaml"}
            ):
                continue
            assert z.read(member) == Path(member).read_bytes(), (
                f"Notebook bundle is stale: {member}"
            )
