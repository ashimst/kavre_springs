"""
In-process executor for Jupyter Notebooks.
Executes code cells sequentially in an isolated namespace, captures stdout, display objects,
and matplotlib plots into native Jupyter mime-type cell outputs, and saves the executed notebook.
"""

import sys, os, io, json, base64
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import pandas as pd
import numpy as np

def execute_notebook(nb_path: str):
    print(f"Executing {nb_path} in-process...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured_figures = []

    def custom_show(*args, **kwargs):
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
            buf.seek(0)
            captured_figures.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close("all")

    def custom_display(*objs):
        for obj in objs:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                print(obj.to_string())
            else:
                print(obj)

    # Isolated execution namespace
    ns = {
        "__file__": str(Path(nb_path).resolve()),
        "__name__": "__main__",
        "display": custom_display,
        "plt": plt
    }
    plt.show = custom_show

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        code = "".join(cell["source"])
        print(f"  Executing cell {idx} ({len(code.splitlines())} lines)...")
        
        stdout_buf = io.StringIO()
        outputs = []
        captured_figures.clear()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
                exec(code, ns)
                
            stdout_text = stdout_buf.getvalue()
            if stdout_text:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": stdout_text.splitlines(keepends=True)
                })

            for img_b64 in captured_figures:
                outputs.append({
                    "output_type": "display_data",
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure size>"]
                    },
                    "metadata": {}
                })

            cell["outputs"] = outputs
            cell["execution_count"] = idx

        except Exception as e:
            import traceback
            err_text = traceback.format_exc()
            print(f"    Error in cell {idx}: {e}")
            outputs.append({
                "name": "stderr",
                "output_type": "stream",
                "text": err_text.splitlines(keepends=True)
            })
            cell["outputs"] = outputs
            cell["execution_count"] = idx
            break

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Finished {nb_path} successfully!")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "notebooks/06_spatial_mining_and_vulnerability.ipynb"
    execute_notebook(target)
