"""Utility script that exports the provided notebook to a plain Python file."""

from __future__ import annotations

import json
from datetime import datetime

from config import EXPORTED_NOTEBOOK_SCRIPT, NOTEBOOK_PATH


def export_notebook(source=NOTEBOOK_PATH, destination=EXPORTED_NOTEBOOK_SCRIPT) -> None:
    """Dump all code cells from the Jupyter notebook into a .py file."""

    nb = json.loads(source.read_text())
    code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]

    header = [
        "# Auto-generated from {}\n".format(source.name),
        "# Exported at {}\n\n".format(datetime.utcnow().isoformat() + "Z"),
    ]

    destination.write_text("".join(header))

    with destination.open("a") as handle:
        for idx, cell in enumerate(code_cells, start=1):
            handle.write(f"# In[{idx}]\n")
            handle.write("".join(cell.get("source", [])))
            handle.write("\n\n")

    print(f"Notebook exported to {destination}")


if __name__ == "__main__":
    export_notebook()
