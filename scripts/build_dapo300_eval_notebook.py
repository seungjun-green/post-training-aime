"""Reuse the AMC/MATH evaluation UI for the fresh minibatch run's checkpoint 300."""

from build_dapo140_eval_notebook import cells as checkpoint140_cells
from build_notebooks import write_notebook


def cells():
    notebook_cells = checkpoint140_cells()
    for cell in notebook_cells:
        cell.source = (cell.source
            .replace("140", "300")
            .replace("from the 100-to-300 continuation", "from the fresh base-model minibatch run")
            .replace("the continuation output folder", "the minibatch training output folder")
            .replace("LG-AIME-DAPO-100to300", "LG-AIME-DAPO-MiniBatch300")
            .replace("LG-AIME-DAPO-Eval-300-AMC-MATH-temp0",
                     "LG-AIME-DAPO-MiniBatch300-Eval-AMC-MATH-temp0"))
    return notebook_cells


if __name__ == "__main__":
    write_notebook("evaluate_dapo_checkpoint300_amc_math.ipynb", cells())
