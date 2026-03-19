import importlib.util, sys, json
from pathlib import Path
from collections import defaultdict

# Patch parent_targets_final to add debug for VA-02
target_id = None

spec = importlib.util.spec_from_file_location("dag", Path("organize-dag-groups.py").resolve())
dag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dag)
SHARED = dag.SHARED

canvas = SHARED.load_canvas("Project.canvas")
tasks = SHARED.get_tasks(canvas)
task_ids = {t.get("id") for t in tasks}
va02_id = next(t["id"] for t in tasks if "VA-02" in t.get("text",""))

# Temporarily patch layout_rows
orig_layout_rows = dag.layout_rows

def patched_layout_rows(*args, **kwargs):
    # Inject debug into the x_pos after final sweep
    result = orig_layout_rows(*args, **kwargs)
    positions, group_color = result
    print(f"VA-02 final x: {positions.get(va02_id, ('?',))[0]}")
    # Find VA cards
    for tid, pos in sorted(positions.items(), key=lambda kv: kv[1][1]):
        node = next((t for t in tasks if t["id"] == tid), None)
        if node and "VA-" in node.get("text",""):
            print(f"  {node['text'].split(chr(10))[0][:30]:30s} x={pos[0]} y={pos[1]}")
    return result

dag.layout_rows = patched_layout_rows

# Run organize
dag.organize("Project.canvas", "Project.dag-groups.canvas")
