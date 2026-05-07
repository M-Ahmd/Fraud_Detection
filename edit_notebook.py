import json

import re

notebook_path = r"c:\projects\AI_final\EDA.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

toc_lines = ["# Table of Contents\n"]
for cell in nb['cells']:
    if cell.get('cell_type') == 'markdown':
        for line in cell.get('source', []):
            match = re.match(r'^(#{1,6})\s+(.*)', line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                # Create a link anchor: lowercase, replace spaces with hyphens, remove non-alphanumeric (simplistic approach)
                anchor = re.sub(r'[^a-zA-Z0-9\s-]', '', title.lower()).replace(' ', '-')
                indent = "  " * (level - 1)
                toc_lines.append(f"{indent}- [{title}](#{anchor})\n")

toc_cell = {
    "cell_type": "markdown",
    "id": "table_of_contents",
    "metadata": {},
    "source": toc_lines
}

# Avoid adding multiple TOCs if run multiple times
if not (nb['cells'][0].get('cell_type') == 'markdown' and 'Table of Contents' in "".join(nb['cells'][0].get('source', []))):
    nb['cells'].insert(0, toc_cell)
else:
    nb['cells'][0] = toc_cell

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with Table of Contents.")

print("Notebook updated successfully.")
