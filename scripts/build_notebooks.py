"""Build the active English data-preparation and evaluation notebooks."""

import argparse
import base64
import io
import textwrap
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source, collapsed=False):
    cell = nbformat.v4.new_code_cell(textwrap.dedent(source).strip())
    if collapsed:
        cell.metadata["cellView"] = "form"
    return cell


def payload(paths):
    content = io.BytesIO()
    with ZipFile(content, "w", ZIP_DEFLATED) as archive:
        for path in paths:
            info = ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return base64.b64encode(content.getvalue()).decode()


def bootstrap(encoded):
    return code(
        f"""
        # @title Install the bundled, tested project code (self-contained notebook)
        import base64, io, sys, zipfile
        from pathlib import Path
        CODE_ROOT = Path("/content/lg-korean-aime")
        CODE_ROOT.mkdir(parents=True, exist_ok=True)
        BUNDLE = {encoded!r}
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as bundle:
            bundle.extractall(CODE_ROOT)
        sys.path.insert(0, str(CODE_ROOT))
        print("Project code ready:", CODE_ROOT)
    """,
        collapsed=True,
    )


def write_notebook(name, cells, selected=None):
    if selected is not None and name != selected:
        return
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.update(
        kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
        language_info={"name": "python"},
        colab={"name": name},
    )
    # Stable cell IDs keep notebook rebuilds reviewable.
    for index, cell in enumerate(nb.cells):
        cell.id = f"cell-{index:02d}"
    nbformat.validate(nb)
    nbformat.write(nb, ROOT / "notebooks" / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        choices=[
            "prepare_english_datasets.ipynb",
            "evaluate_baseline_english.ipynb",
        ],
    )
    args = parser.parse_args()
    if args.notebook in {None, "evaluate_baseline_english.ipynb"}:
        from english_eval_notebook import english_eval_cells

        write_notebook("evaluate_baseline_english.ipynb", english_eval_cells(markdown, code))
        if args.notebook == "evaluate_baseline_english.ipynb":
            return
    if args.notebook in {None, "prepare_english_datasets.ipynb"}:
        from english_notebook import english_cells

        bundle = payload(
            [
                ROOT / p
                for p in [
                    "common/__init__.py",
                    "common/io.py",
                    "pipeline/__init__.py",
                    "pipeline/datasets.py",
                    "pipeline/decontamination.py",
                    "pipeline/english_publish.py",
                    "configs/datasets.yaml",
                ]
            ]
        )
        write_notebook(
            "prepare_english_datasets.ipynb", english_cells(ROOT, bundle, markdown, code, bootstrap)
        )
        if args.notebook == "prepare_english_datasets.ipynb":
            return


if __name__ == "__main__":
    main()
