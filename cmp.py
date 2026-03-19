import json, sys, pathlib
import importlib.util
from collections import defaultdict

spec = importlib.util.spec_from_file_location("dag", pathlib.Path("organize-dag-groups.py").resolve())
dag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dag)
SHARED = dag.SHARED

canvas  = SHARED.load_canvas("layout-fixtures/stress-test.canvas")
tasks   = SHARED.get_tasks(canvas)
task_ids = {t.get("id") for t in tasks}
key_to_id = {t["text"].split("\n")[0][3:9].strip(): t["id"] for t in tasks}
id_to_key = {v: k for k, v in key_to_id.items()}

incoming_g, outgoing_g = dag.build_graph(canvas, task_ids)
depths = dag.compute_depths(task_ids, incoming_g, outgoing_g)
reduced_out, reduced_in = dag.transitive_reduction(task_ids, outgoing_g)
membership = dag.infer_membership(canvas, task_ids)

WATCH = {key_to_id[k] for k in ["INF-03", "BE-01", "BE-02", "BE-03", "BE-04"]}

# Patch place_row_in_column inside layout_rows by patching x_pos assignment
# We do this by monkey-patching the function temporarily
orig = dag.layout_rows

def patched_layout_rows(tasks, depths, incoming, outgoing, membership=None, **kw):
    # Run original then read x_pos from closure using positions
    result = orig(tasks, depths, incoming, outgoing, membership=membership, **kw)
    positions, group_color = result
    print("Final positions for watched cards:")
    for tid in WATCH:
        key = id_to_key[tid]
        x, y, w, h = positions[tid]
        print(f"  {key}: x={x}")
    return result

dag.layout_rows = patched_layout_rows

# Now run with prints inside - we need to trace DURING execution
# Use a fresh reimplementation with traces:
from collections import defaultdict
import math

def layout_rows_traced(tasks, depths, incoming, outgoing, membership=None,
                       row_gap=70, layer_gap=250, group_gap=200, margin_x=80, margin_y=80):
    membership = membership or {}
    rows = defaultdict(list)
    for task in tasks:
        rows[depths[task.get("id")]].append(task)
    for depth in rows:
        rows[depth].sort(key=lambda t: (SHARED.task_id_str(t) or "", t.get("id", "")))

    sizes = dag.uniform_sizes(tasks)
    sorted_depths = sorted(rows.keys())

    group_depths = defaultdict(set)
    for t in tasks:
        gid = membership.get(t.get("id"))
        if gid is not None:
            group_depths[gid].add(depths[t.get("id")])

    group_ids_ordered = []
    for depth in sorted_depths:
        for t in rows[depth]:
            gid = membership.get(t.get("id"))
            if gid is not None and gid not in group_ids_ordered:
                group_ids_ordered.append(gid)

    group_depth_range = {gid: (min(ds), max(ds)) for gid, ds in group_depths.items()}
    conflicts = defaultdict(set)
    for i, ga in enumerate(group_ids_ordered):
        a_lo, a_hi = group_depth_range[ga]
        for gb in group_ids_ordered[i + 1:]:
            b_lo, b_hi = group_depth_range[gb]
            if a_lo <= b_hi and b_lo <= a_hi:
                conflicts[ga].add(gb)
                conflicts[gb].add(ga)

    group_color = {}
    for gid in group_ids_ordered:
        used = {group_color[nb] for nb in conflicts[gid] if nb in group_color}
        color = 0
        while color in used:
            color += 1
        group_color[gid] = color

    print("Group colors:", {gid.replace("group-","")[:6]: c for gid,c in group_color.items()})

    num_cols = max(group_color.values(), default=-1) + 1
    UNGROUPED_COL = num_cols
    def col_of(tid):
        gid = membership.get(tid)
        return group_color[gid] if gid is not None else UNGROUPED_COL

    col_row_widths = defaultdict(lambda: defaultdict(int))
    for depth in sorted_depths:
        col_cards = defaultdict(list)
        for t in rows[depth]:
            col_cards[col_of(t.get("id"))].append(t.get("id"))
        for col, tids in col_cards.items():
            total_w = sum(sizes[tid][0] for tid in tids) + row_gap * max(0, len(tids) - 1)
            col_row_widths[col][depth] = total_w

    col_widths = {}
    for col in sorted(set(col_of(t.get("id")) for t in tasks)):
        col_widths[col] = max(col_row_widths[col].values()) if col_row_widths[col] else 0
    print("Col widths:", col_widths)

    col_x_start = {}
    x_cursor = margin_x
    for col in sorted(col_widths.keys()):
        col_x_start[col] = x_cursor
        x_cursor += col_widths[col] + group_gap
    print("Col x starts:", col_x_start)

    x_pos = {}
    def card_center(tid): return x_pos[tid] + sizes[tid][0] / 2

    def place_row_in_column(col, tids, targets):
        if not tids: return tids
        sorted_ids = sorted(tids, key=lambda tid: targets.get(tid, 0))
        col_start = col_x_start[col]
        x = col_start
        for tid in sorted_ids:
            w = sizes[tid][0]
            target = targets.get(tid, x + w / 2)
            left = max(x, max(col_start, int(target - w / 2)))
            if tid in WATCH:
                print(f"  place {id_to_key[tid]}: target={target:.1f} left={left}  (x_cursor={x})")
            x_pos[tid] = left
            x = left + w + row_gap
        return sorted_ids

    def parent_targets(depth):
        targets = {}
        for t in rows[depth]:
            tid = t.get("id"); my_col = col_of(tid)
            parents = [p for p in incoming.get(tid, set()) if p in x_pos and col_of(p) == my_col]
            if parents: targets[tid] = sum(card_center(p) for p in parents) / len(parents)
            else: targets[tid] = x_pos.get(tid, col_x_start.get(my_col, margin_x)) + sizes[tid][0] / 2
        return targets

    def child_targets(depth):
        targets = {}
        for t in rows[depth]:
            tid = t.get("id"); my_col = col_of(tid)
            children = [c for c in outgoing.get(tid, set()) if c in x_pos and col_of(c) == my_col]
            targets[tid] = sum(card_center(c) for c in children) / len(children) if children else card_center(tid)
        return targets

    def place_depth(depth, targets):
        col_cards = defaultdict(list)
        for t in rows[depth]:
            col_cards[col_of(t.get("id"))].append(t.get("id"))
        order_ids = []
        for col in sorted(col_cards.keys()):
            order_ids.extend(place_row_in_column(col, col_cards[col], targets))
        return order_ids

    order = {}
    for depth in sorted_depths:
        if depth == sorted_depths[0]:
            targets = {t.get("id"): col_x_start.get(col_of(t.get("id")), margin_x) + sizes[t.get("id")][0] / 2 for t in rows[depth]}
        else:
            targets = parent_targets(depth)
        order[depth] = place_depth(depth, targets)

    for _ in range(4):
        for depth in reversed(sorted_depths[:-1]):
            order[depth] = place_depth(depth, child_targets(depth))
        for depth in sorted_depths[1:]:
            order[depth] = place_depth(depth, parent_targets(depth))

    if x_pos and membership:
        for depth in sorted_depths:
            col_cards = defaultdict(list)
            for t in rows[depth]:
                col_cards[col_of(t.get("id"))].append(t.get("id"))
            for col, tids in col_cards.items():
                col_start = col_x_start[col]
                min_card_x = min(x_pos[tid] for tid in tids)
                shift = min_card_x - col_start
                if shift != 0:
                    for tid in tids:
                        x_pos[tid] -= shift

    print("\nAfter compaction:")
    for k in ["INF-01", "INF-02", "INF-03", "BE-01", "BE-02", "BE-03"]:
        print(f"  {k}: x_pos={x_pos[key_to_id[k]]}")

    def parent_targets_final(depth):
        slot_counts = defaultdict(int)
        for t in rows[depth]:
            slot_counts[col_of(t.get("id"))] += 1
        targets = {}
        for t in rows[depth]:
            tid = t.get("id"); my_col = col_of(tid)
            parents = [p for p in incoming.get(tid, set()) if p in x_pos and col_of(p) == my_col]
            if parents:
                targets[tid] = sum(card_center(p) for p in parents) / len(parents)
            elif membership and slot_counts[my_col] == 1:
                gid = membership.get(tid)
                same_group = [o for o in x_pos if membership.get(o) == gid and o != tid]
                if same_group:
                    targets[tid] = sum(card_center(o) for o in same_group) / len(same_group)
                    continue
            targets[tid] = x_pos.get(tid, col_x_start.get(my_col, margin_x)) + sizes[tid][0] / 2
        return targets

    print("\nparentTargetsFinal pass:")
    for depth in sorted_depths[1:]:
        targets = parent_targets_final(depth)
        watched_in_depth = [id_to_key[t.get("id")] for t in rows[depth] if t.get("id") in WATCH]
        if watched_in_depth:
            print(f"  depth {depth}: processing {watched_in_depth}")
            for t in rows[depth]:
                tid = t.get("id")
                if tid in WATCH:
                    print(f"    {id_to_key[tid]}: target={targets.get(tid,'?'):.1f if isinstance(targets.get(tid), float) else targets.get(tid,'?')}")
        order[depth] = place_depth(depth, targets)
        if watched_in_depth:
            for k in watched_in_depth:
                print(f"    {k} placed at: {x_pos[key_to_id[k]]}")

layout_rows_traced(tasks, depths, reduced_in, reduced_out, membership=membership)
