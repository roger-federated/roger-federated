---
name: ipynb
description: Read, create, edit, and run Jupyter notebooks (.ipynb). Use whenever a task touches a .ipynb file; do not hand-edit notebooks with literal find/replace.
---

A `.ipynb` is a single JSON document (cells, outputs, metadata, schema version). Editing it
with the literal-string `edit_file` tool corrupts the JSON or the cell structure easily, so
drive it with the `nbformat` library instead: write small Python and run it via `run_command`.

Setup (once): `uv pip install nbformat`  (add `nbconvert jupyter` to execute notebooks).

Read:
```python
import nbformat
nb = nbformat.read("path.ipynb", as_version=4)
for i, c in enumerate(nb.cells):
    print(i, c.cell_type, repr(c.source[:80]))
```

Edit an existing cell (locate by index or by matching its source), reassign `.source`, write back:
```python
nb.cells[3].source = "new code"
nbformat.write(nb, "path.ipynb")
```

Add a cell:
```python
nb.cells.insert(2, nbformat.v4.new_code_cell("import pandas as pd"))
# or new_markdown_cell("# Title"); then nbformat.write(nb, path)
```

Execute (runs every cell, saves outputs in place):
```bash
jupyter nbconvert --to notebook --execute --inplace path.ipynb
```
Reading a cell's results: inspect `cell.outputs`; stream text is `output["text"]`, rich
results are `output["data"]["text/plain"]`.

Before committing a notebook, strip noisy execution outputs:
`jupyter nbconvert --clear-output --inplace path.ipynb`.
