#!/usr/bin/env python3
"""Simple DAG layout: assign each card a depth (longest path from root), place in rows."""

from __future__ import annotations

import argparse
import copy
import importlib.util
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Load shared helpers from canvas-tool.py
# ---------------------------------------------------------------------------

def _load_shared():
    tool_path = Path(__file__).resolve().with_name("canvas-tool.py")
    spec = importlib.util.spec_from_file_location("canvas_tool_shared", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared helpers from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHARED = _load_shared()


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def build_graph(canvas, task_ids):
    """Return (incoming, outgoing) adjacency dicts for task-to-task edges."""
    incoming = {tid: set() for tid in task_ids}
    outgoing = {tid: set() for tid in task_ids}
    for edge in canvas.get("edges", []):
        f, t = edge.get("fromNode"), edge.get("toNode")
        if f in task_ids and t in task_ids:
            outgoing[f].add(t)
            incoming[t].add(f)
    return incoming, outgoing


def compute_depths(task_ids, incoming, outgoing):
    """Longest-path depth from roots. Depth 0 = no parents."""
    depths = {tid: 0 for tid in task_ids}
    indegree = {tid: len(incoming[tid]) for tid in task_ids}
    queue = [tid for tid, deg in indegree.items() if deg == 0]
    visited = 0

    while queue:
        tid = queue.pop(0)
        visited += 1
        for child in outgoing[tid]:
            depths[child] = max(depths[child], depths[tid] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if visited != len(task_ids):
        raise RuntimeError("Cycle detected in task dependencies — cannot compute depths.")

    return depths


def transitive_reduction(task_ids, outgoing):
    """Remove redundant edges: if A->B->C exists, drop A->C.

    Returns (reduced_outgoing, reduced_incoming) with only essential edges.
    """
    reduced_out = {}
    for tid in task_ids:
        indirect = set()
        for child in outgoing[tid]:
            stack = list(outgoing[child])
            while stack:
                node = stack.pop()
                if node not in indirect:
                    indirect.add(node)
                    stack.extend(outgoing[node])
        reduced_out[tid] = outgoing[tid] - indirect

    # Build reduced incoming from reduced outgoing
    reduced_in = {tid: set() for tid in task_ids}
    for tid, children in reduced_out.items():
        for child in children:
            reduced_in[child].add(tid)

    return reduced_out, reduced_in


# ---------------------------------------------------------------------------
# Card sizing (simple heuristic)
# ---------------------------------------------------------------------------

def estimate_card_size(node):
    """Return (width, height) based on text length."""
    text = node.get("text", "")
    lines = text.split("\n")
    longest_line = max((len(line) for line in lines), default=0)
    width = max(260, min(420, int(longest_line * 8) + 52))
    height = max(180, len(lines) * 24 + 60)
    return width, height


def uniform_sizes(tasks):
    """Compute sizes for all tasks, then set all widths to the max."""
    raw = {t.get("id"): estimate_card_size(t) for t in tasks}
    max_w = max(w for w, _ in raw.values())
    return {tid: (max_w, h) for tid, (_, h) in raw.items()}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout_rows(tasks, depths, incoming, outgoing, membership=None,
                row_gap=70, layer_gap=250, group_gap=200, margin_x=80, margin_y=80,
                horizontal=False):
    """Place cards in rows by depth, with group columns to avoid overlapping groups.

    Each group gets a dedicated x-column. Within each column, cards are placed
    by depth (y) and aligned to parent/child targets (x) via sweeps. Group
    columns are sized to the widest row across all depths and separated by
    group_gap, ensuring bounding boxes never overlap.
    """
    membership = membership or {}

    rows = defaultdict(list)
    for task in tasks:
        rows[depths[task.get("id")]].append(task)

    for depth in rows:
        rows[depth].sort(key=lambda t: (SHARED.task_id_str(t) or "", t.get("id", "")))

    sizes = uniform_sizes(tasks)
    # In horizontal mode, swap w/h for layout so spacing matches transposed axes
    if horizontal:
        sizes = {tid: (h, w) for tid, (w, h) in sizes.items()}
    sorted_depths = sorted(rows.keys())

    # ----- Group column assignment via conflict graph -----
    # Two groups conflict only if they share tasks at the same depth.
    # Non-conflicting groups share an x-column (they occupy different rows).
    group_depths = defaultdict(set)
    for t in tasks:
        gid = membership.get(t.get("id"))
        if gid is not None:
            group_depths[gid].add(depths[t.get("id")])

    # Ordered by first appearance
    group_ids_ordered = []
    for depth in sorted_depths:
        for t in rows[depth]:
            gid = membership.get(t.get("id"))
            if gid is not None and gid not in group_ids_ordered:
                group_ids_ordered.append(gid)

    # Build conflict edges: groups conflict if their depth RANGES overlap
    # (not just exact depths), since bounding boxes span min..max depth.
    group_depth_range = {gid: (min(ds), max(ds)) for gid, ds in group_depths.items()}
    conflicts = defaultdict(set)
    for i, ga in enumerate(group_ids_ordered):
        a_lo, a_hi = group_depth_range[ga]
        for gb in group_ids_ordered[i + 1:]:
            b_lo, b_hi = group_depth_range[gb]
            if a_lo <= b_hi and b_lo <= a_hi:  # ranges overlap
                conflicts[ga].add(gb)
                conflicts[gb].add(ga)

    # Greedy graph coloring (groups ordered by first appearance)
    group_color = {}
    for gid in group_ids_ordered:
        used = {group_color[nb] for nb in conflicts[gid] if nb in group_color}
        color = 0
        while color in used:
            color += 1
        group_color[gid] = color

    num_cols = max(group_color.values(), default=-1) + 1
    UNGROUPED_COL = num_cols

    def col_of(tid):
        gid = membership.get(tid)
        if gid is not None:
            return group_color[gid]
        return UNGROUPED_COL

    # Compute column widths: for each column, the widest row it occupies
    col_row_widths = defaultdict(lambda: defaultdict(int))  # col -> depth -> total width
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

    # Assign column x-offsets
    col_x_start = {}
    x_cursor = margin_x
    for col in sorted(col_widths.keys()):
        col_x_start[col] = x_cursor
        x_cursor += col_widths[col] + group_gap

    # ----- Place cards within columns -----
    x_pos = {}

    def card_center(tid):
        return x_pos[tid] + sizes[tid][0] / 2

    def place_row_in_column(col, tids, targets):
        """Place cards within a column, respecting targets but staying in bounds."""
        if not tids:
            return tids
        sorted_ids = sorted(tids, key=lambda tid: targets.get(tid, 0))
        col_start = col_x_start[col]
        x = col_start
        for tid in sorted_ids:
            w = sizes[tid][0]
            target = targets.get(tid, x + w / 2)
            # Clamp to column bounds
            left = max(x, max(col_start, int(target - w / 2)))
            x_pos[tid] = left
            x = left + w + row_gap
        return sorted_ids

    def parent_targets(depth):
        """Target x = average of same-column parents' centers (cross-column ignored)."""
        targets = {}
        for t in rows[depth]:
            tid = t.get("id")
            my_col = col_of(tid)
            parents = [p for p in incoming.get(tid, set()) if p in x_pos and col_of(p) == my_col]
            if parents:
                targets[tid] = sum(card_center(p) for p in parents) / len(parents)
            else:
                targets[tid] = x_pos.get(tid, col_x_start.get(my_col, margin_x)) + sizes[tid][0] / 2
        return targets

    def child_targets(depth):
        """Target x = average of same-column children's centers (cross-column ignored)."""
        targets = {}
        for t in rows[depth]:
            tid = t.get("id")
            my_col = col_of(tid)
            children = [c for c in outgoing.get(tid, set()) if c in x_pos and col_of(c) == my_col]
            if children:
                targets[tid] = sum(card_center(c) for c in children) / len(children)
            else:
                targets[tid] = card_center(tid)
        return targets

    def place_depth(depth, targets):
        """Place all cards at a depth, column by column."""
        col_cards = defaultdict(list)
        for t in rows[depth]:
            col_cards[col_of(t.get("id"))].append(t.get("id"))
        order_ids = []
        for col in sorted(col_cards.keys()):
            order_ids.extend(place_row_in_column(col, col_cards[col], targets))
        return order_ids

    # Initial top-down pass
    order = {}
    for depth in sorted_depths:
        if depth == sorted_depths[0]:
            # First row: simple left-to-right within columns
            targets = {t.get("id"): col_x_start.get(col_of(t.get("id")), margin_x) + sizes[t.get("id")][0] / 2
                       for t in rows[depth]}
        else:
            targets = parent_targets(depth)
        order[depth] = place_depth(depth, targets)

    # Refine with alternating bottom-up / top-down sweeps
    for _ in range(4):
        for depth in reversed(sorted_depths[:-1]):
            targets = child_targets(depth)
            order[depth] = place_depth(depth, targets)

        for depth in sorted_depths[1:]:
            targets = parent_targets(depth)
            order[depth] = place_depth(depth, targets)

    # Compact: when using group columns, align each row within its column
    # so the leftmost card sits at col_x_start (consistent across depths).
    if x_pos and membership:
        for depth in sorted_depths:
            col_cards = defaultdict(list)
            for t in rows[depth]:
                col_cards[col_of(t.get("id"))].append(t.get("id"))
            for col, tids in col_cards.items():
                if not tids:
                    continue
                col_start = col_x_start[col]
                min_card_x = min(x_pos[tid] for tid in tids)
                shift = min_card_x - col_start
                if shift != 0:
                    for tid in tids:
                        x_pos[tid] -= shift
    elif x_pos:
        shift = min(x_pos.values()) - margin_x
        if shift != 0:
            for tid in x_pos:
                x_pos[tid] -= shift

    # Final top-down pass: restore parent-child alignment broken by compaction.
    # Singletons (alone in their col×depth) with no same-col parents use the
    # average of already-placed same-group members so they align with the group.
    def parent_targets_final(depth):
        slot_counts = defaultdict(int)
        for t in rows[depth]:
            slot_counts[col_of(t.get("id"))] += 1
        targets = {}
        for t in rows[depth]:
            tid = t.get("id")
            my_col = col_of(tid)
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

    for depth in sorted_depths[1:]:
        targets = parent_targets_final(depth)
        order[depth] = place_depth(depth, targets)

    # Build final positions with y
    positions = {}
    y_cursor = margin_y
    # Restore real card dimensions for final positions
    real_sizes = uniform_sizes(tasks)
    for depth in sorted_depths:
        row_ids = order[depth]
        row_height = max(sizes[tid][1] for tid in row_ids)
        for tid in row_ids:
            w, h = real_sizes[tid]
            positions[tid] = (x_pos[tid], y_cursor + (row_height - sizes[tid][1]) // 2, w, h)
        y_cursor += row_height + layer_gap

    return positions, group_color


# ---------------------------------------------------------------------------
# Group membership & bounding boxes
# ---------------------------------------------------------------------------

def infer_membership(canvas, task_ids):
    """Map each task to its containing group (from original canvas geometry).

    Returns {task_id: group_id} for tasks inside a group.
    """
    membership = {}
    for node in canvas.get("nodes", []):
        if node.get("id") not in task_ids:
            continue
        group = SHARED.get_group_for_node(canvas, node)
        if group is not None:
            membership[node.get("id")] = group.get("id")
    return membership


def compute_group_bounds(positions, membership, pad_x=60, pad_top=40, pad_bottom=40):
    """Compute bounding box for each group from its members' final positions.

    Each group wraps tightly around its own cards with padding.
    Returns {group_id: (x, y, width, height)}.
    """
    group_members = defaultdict(list)
    for tid, gid in membership.items():
        if tid in positions:
            group_members[gid].append(tid)

    bounds = {}
    for gid, members in group_members.items():
        min_x = min(positions[tid][0] for tid in members)
        min_y = min(positions[tid][1] for tid in members)
        max_x = max(positions[tid][0] + positions[tid][2] for tid in members)
        max_y = max(positions[tid][1] + positions[tid][3] for tid in members)
        bounds[gid] = (
            min_x - pad_x, min_y - pad_top,
            (max_x - min_x) + 2 * pad_x,
            (max_y - min_y) + pad_top + pad_bottom,
        )
    return bounds


# ---------------------------------------------------------------------------
# Apply layout
# ---------------------------------------------------------------------------

def apply_layout(canvas, positions, reduced_outgoing, keep_groups=True, horizontal=False):
    """Update node positions, set ports, refit groups, filter redundant edges."""
    updated = copy.deepcopy(canvas)

    task_ids = set(positions.keys())

    # Transpose for horizontal mode: swap x/y axes on card positions
    if horizontal:
        positions = {tid: (y, x, w, h) for tid, (x, y, w, h) in positions.items()}

    # Compute group bounds AFTER transpose so padding is applied correctly.
    # Each group wraps tightly around its own cards (no column alignment needed
    # since same-column groups occupy different depth ranges and never overlap).
    if keep_groups:
        membership = infer_membership(canvas, task_ids)
        group_bounds = compute_group_bounds(positions, membership)
    else:
        group_bounds = {}

    # Remove groups that have no members; keep those with members (if keep_groups)
    new_nodes = []
    for n in updated.get("nodes", []):
        if n.get("type") == "group":
            gid = n.get("id")
            if keep_groups and gid in group_bounds:
                x, y, w, h = group_bounds[gid]
                n["x"] = int(x)
                n["y"] = int(y)
                n["width"] = int(w)
                n["height"] = int(h)
                new_nodes.append(n)
        else:
            new_nodes.append(n)
    updated["nodes"] = new_nodes

    # Keep only essential edges (transitive reduction) that reference valid nodes
    valid_ids = {n.get("id") for n in updated["nodes"]}
    updated["edges"] = [
        e for e in updated.get("edges", [])
        if e.get("fromNode") in valid_ids
        and e.get("toNode") in valid_ids
        and e.get("toNode") in reduced_outgoing.get(e.get("fromNode"), set())
    ]

    # Apply positions to task cards
    for node in updated["nodes"]:
        tid = node.get("id")
        if tid in positions:
            x, y, w, h = positions[tid]
            node["x"] = x
            node["y"] = y
            node["width"] = w
            node["height"] = h

    # Set edge ports based on flow direction
    if horizontal:
        from_side, to_side = "right", "left"
    else:
        from_side, to_side = "bottom", "top"
    for edge in updated["edges"]:
        edge["fromSide"] = from_side
        edge["toSide"] = to_side

    return updated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def organize(canvas_path, output_path, layer_gap=250, row_gap=70, keep_groups=True, horizontal=False):
    canvas = SHARED.load_canvas(canvas_path)
    tasks = SHARED.get_tasks(canvas)
    if not tasks:
        raise RuntimeError("No task cards found in canvas.")

    task_ids = {t.get("id") for t in tasks}
    incoming, outgoing = build_graph(canvas, task_ids)
    depths = compute_depths(task_ids, incoming, outgoing)
    reduced_out, reduced_in = transitive_reduction(task_ids, outgoing)

    original_edges = sum(len(children) for children in outgoing.values())
    reduced_edges = sum(len(children) for children in reduced_out.values())

    # Infer group membership from original canvas geometry
    membership = infer_membership(canvas, task_ids) if keep_groups else {}

    # Use reduced edges for layout so targets reflect direct deps only
    positions, _group_col = layout_rows(tasks, depths, reduced_in, reduced_out,
                                        membership=membership, row_gap=row_gap, layer_gap=layer_gap,
                                        horizontal=horizontal)
    updated = apply_layout(canvas, positions, reduced_out, keep_groups=keep_groups,
                           horizontal=horizontal)

    SHARED.save_canvas(output_path, updated)

    max_depth = max(depths.values(), default=0)
    removed = original_edges - reduced_edges
    groups_kept = len([n for n in updated["nodes"] if n.get("type") == "group"])
    print(f"Tasks: {len(tasks)}")
    print(f"Edges: {original_edges} -> {reduced_edges} ({removed} redundant removed)")
    print(f"Groups: {groups_kept}")
    print(f"Rows (depth 0..{max_depth}): {max_depth + 1}")
    for d in range(max_depth + 1):
        count = sum(1 for v in depths.values() if v == d)
        print(f"  depth {d}: {count} cards")
    print(f"Wrote: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple DAG row layout by depth.")
    parser.add_argument("canvas_file", help="Input .canvas file")
    parser.add_argument("--output", help="Output path (default: <input>.dag-groups.canvas)")
    parser.add_argument("--layer-gap", type=int, default=250, help="Space between dependency levels")
    parser.add_argument("--row-gap", type=int, default=70, help="Horizontal gap between cards")
    parser.add_argument("--no-groups", action="store_true", help="Strip group nodes from output")
    parser.add_argument("--horizontal", action="store_true", help="Left-to-right layout instead of top-to-bottom")
    args = parser.parse_args()

    output = args.output or str(
        Path(args.canvas_file).with_name(
            f"{Path(args.canvas_file).stem}.dag-groups{Path(args.canvas_file).suffix}"
        )
    )
    organize(args.canvas_file, output, layer_gap=args.layer_gap, row_gap=args.row_gap,
             keep_groups=not args.no_groups, horizontal=args.horizontal)


if __name__ == "__main__":
    main()
