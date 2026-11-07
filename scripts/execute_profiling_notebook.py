"""Run the actual notebook in a fresh headless IPython process and save outputs.

No global kernelspec installation, local socket server, or synthetic output.
Tables and PNGs come from IPython's real rich-display publisher.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config

ROOT = Path(__file__).resolve().parents[1]


class HeadlessNotebookShell(InteractiveShell):
    """Inline rendering has no native GUI event loop to enable."""

    def enable_gui(self, gui=None):
        if gui not in (None, "inline"):
            raise ValueError("This notebook executor supports inline figures only")


def save(path, notebook):
    # Atomic replacement keeps viewers from seeing a partially written notebook.
    temporary = path.with_suffix(".ipynb.tmp")
    payload = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    for attempt in range(4):
        try:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)
            return
        except OSError:
            if attempt == 3:
                raise
            time.sleep(0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/01_data_profiling.ipynb")
    args = parser.parse_args()
    path = args.notebook.resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"], cell["outputs"] = None, []
    notebook["metadata"].pop("execution", None)
    config = Config()
    config.HistoryManager.hist_file = ":memory:"
    shell = HeadlessNotebookShell.instance(config=config)
    shell.display_formatter.active_types = ["text/plain", "text/html", "text/markdown", "image/png", "image/svg+xml"]
    start = time.monotonic()
    executed = 0
    os.chdir(path.parent)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        print(f"Executing code cell {executed + 1}, notebook cell {index + 1}/{len(notebook['cells'])}", flush=True)
        cell["execution_count"] = shell.execution_count
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(source, store_history=True)
        if captured.stdout:
            cell["outputs"].append({"output_type": "stream", "name": "stdout", "text": captured.stdout})
        if captured.stderr:
            cell["outputs"].append({"output_type": "stream", "name": "stderr", "text": captured.stderr})
        for output in captured.outputs:
            cell["outputs"].append({"output_type": "display_data", "data": output.data, "metadata": output.metadata})
        failure = result.error_before_exec or result.error_in_exec
        if failure is not None:
            cell["outputs"].append({"output_type": "error", "ename": type(failure).__name__, "evalue": str(failure),
                                    "traceback": traceback.format_exception(type(failure), failure, failure.__traceback__)})
            notebook["metadata"]["execution"] = {"status": "failed", "code_cells_completed": executed, "backend": "IPython"}
            save(path, notebook)
            print(captured.stdout, flush=True)
            raise RuntimeError(f"Notebook failed in cell {index + 1}") from failure
        executed += 1
        save(path, notebook)
    errors = [o for c in notebook["cells"] if c["cell_type"] == "code" for o in c["outputs"] if o["output_type"] == "error"]
    images = sum("image/png" in o.get("data", {}) for c in notebook["cells"] for o in c.get("outputs", []))
    assert not errors and images >= 12
    notebook["metadata"]["execution"] = {"status": "completed", "backend": "IPython", "code_cells": executed,
                                           "elapsed_seconds": round(time.monotonic() - start, 2), "inline_pngs": images}
    save(path, notebook)
    print(json.dumps({"status": "completed", "code_cells": executed, "inline_pngs": images,
                      "seconds": round(time.monotonic() - start, 2), "bytes": path.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
