#!/usr/bin/env python3
"""
Standalone sandbox CLI for experimental canvas organization.

This tool intentionally lives outside `canvas-tool.py` so DAG layout work can
iterate without changing the main workflow CLI. It only rewrites geometry and
edge attachment sides; it does not change task text, state, or dependencies.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import importlib.util
import itertools
import math
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def load_shared_module():
    tool_path = Path(__file__).resolve().with_name("canvas-tool.py")
    spec = importlib.util.spec_from_file_location("canvas_tool_shared", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared helpers from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHARED = load_shared_module()
SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_INDEX = {side: index for index, side in enumerate(SIDE_ORDER)}
SIDE_NORMALS = {
    "top": (0.0, -1.0),
    "right": (1.0, 0.0),
    "bottom": (0.0, 1.0),
    "left": (-1.0, 0.0),
}


@dataclass(frozen=True)
class LayoutConfig:
    group_gap: int = 180
    group_band_step: int = 420
    group_width_gap_ratio: float = 0.2
    group_padding_x: int = 60
    group_padding_top: int = 90
    group_padding_bottom: int = 60
    task_gap: int = 40
    level_gap: int = 30
    min_card_width: int = 260
    max_card_width: int = 420
    min_card_height: int = 180
    preferred_max_card_height: int = 420
    min_group_width: int = 380
    min_group_height: int = 260
    card_padding_x: int = 26
    card_padding_top: int = 28
    card_padding_bottom: int = 24
    heading_body_gap: int = 16
    heading_line_height: int = 28
    body_line_height: int = 22
    heading_char_width: float = 8.8
    body_char_width: float = 7.2
    width_step: int = 20
    target_card_aspect: float = 0.82
    max_columns_per_level: int = 2
    long_edge_lane_padding: int = 100
    group_phase_step: int = 140
    max_group_phase_steps: int = 4
    inter_group_corridor_margin: int = 22
    edge_port_gap: int = 7
    edge_handle_min: int = 70
    edge_handle_max: int = 150
    edge_curve_samples: int = 16
    edge_occlusion_margin: int = 18


def stable_task_key(node):
    task_id = SHARED.task_id_str(node) or ""
    return (node.get("y", 0), node.get("x", 0), task_id, node.get("id", ""))


def stable_group_key(group):
    return (
        group.get("x", 0),
        group.get("y", 0),
        group.get("label", ""),
        group.get("id", ""),
    )


def build_task_graph(canvas, task_nodes):
    task_ids = {node.get("id") for node in task_nodes}
    incoming = {task_id: set() for task_id in task_ids}
    outgoing = {task_id: set() for task_id in task_ids}

    for edge in canvas.get("edges", []):
        from_id = edge.get("fromNode")
        to_id = edge.get("toNode")
        if from_id in task_ids and to_id in task_ids:
            outgoing[from_id].add(to_id)
            incoming[to_id].add(from_id)

    return incoming, outgoing


def longest_path_levels(task_nodes, incoming, outgoing):
    order = {
        node.get("id"): index
        for index, node in enumerate(sorted(task_nodes, key=stable_task_key))
    }
    indegree = {task_id: len(parents) for task_id, parents in incoming.items()}
    levels = {task_id: 0 for task_id in order}
    heap = [(order[task_id], task_id) for task_id, degree in indegree.items() if degree == 0]
    heapq.heapify(heap)
    visited = 0

    while heap:
        _, task_id = heapq.heappop(heap)
        visited += 1
        for child_id in sorted(outgoing[task_id], key=lambda item: order[item]):
            levels[child_id] = max(levels[child_id], levels[task_id] + 1)
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                heapq.heappush(heap, (order[child_id], child_id))

    if visited != len(order):
        raise RuntimeError("Task dependency graph is cyclic; organization requires a DAG")

    return levels


def split_card_text(node):
    text = node.get("text", "")
    lines = text.splitlines()
    if not lines:
        return "", ""

    heading = lines[0]
    if heading.startswith("## "):
        heading = heading[3:]

    body = "\n".join(lines[1:]).strip()
    return heading.strip(), body


def wrap_text_block(text, width_chars):
    if not text:
        return []

    wrapped_lines = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if wrapped_lines and wrapped_lines[-1] != "":
                wrapped_lines.append("")
            continue
        wrapped = textwrap.wrap(
            stripped,
            width=max(8, width_chars),
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped_lines.extend(wrapped or [""])
    return wrapped_lines


def longest_token_length(*texts):
    longest = 0
    for text in texts:
        for token in re.findall(r"\S+", text):
            longest = max(longest, len(token))
    return longest


def estimate_card_size(node, config):
    heading, body = split_card_text(node)
    longest_token = longest_token_length(heading, body)
    best_choice = None

    for width in range(config.min_card_width, config.max_card_width + 1, config.width_step):
        usable_width = max(80, width - (2 * config.card_padding_x))
        heading_chars = max(14, int(usable_width / config.heading_char_width))
        body_chars = max(16, int(usable_width / config.body_char_width))

        heading_lines = wrap_text_block(heading, heading_chars)
        body_lines = wrap_text_block(body, body_chars)

        required_height = config.card_padding_top + (len(heading_lines) * config.heading_line_height)
        if body_lines:
            required_height += config.heading_body_gap + (len(body_lines) * config.body_line_height)
        required_height += config.card_padding_bottom
        height = max(config.min_card_height, required_height)

        aspect = width / max(height, 1)
        aspect_penalty = abs(aspect - config.target_card_aspect) * 180
        width_penalty = (width - config.min_card_width) * 0.12
        overflow_penalty = max(0, height - config.preferred_max_card_height) * 1.4
        break_penalty = max(0, longest_token - body_chars) * 9

        score = aspect_penalty + width_penalty + overflow_penalty + break_penalty
        choice = (score, width, height)
        if best_choice is None or choice < best_choice:
            best_choice = choice

    _, width, height = best_choice
    return int(width), int(height)


def infer_group_membership(canvas, task_nodes):
    members = defaultdict(list)
    ungrouped = []
    for node in task_nodes:
        group = SHARED.get_group_for_node(canvas, node)
        if group is None:
            ungrouped.append(node)
            continue
        members[group.get("id")].append(node)
    return members, ungrouped


def build_group_dependency_weights(ordered_groups, members_by_group, outgoing):
    task_to_group = {}
    for group in ordered_groups:
        group_id = group.get("id")
        for node in members_by_group.get(group_id, []):
            task_to_group[node.get("id")] = group_id

    weights = defaultdict(int)
    for from_task, children in outgoing.items():
        from_group = task_to_group.get(from_task)
        if from_group is None:
            continue
        for to_task in children:
            to_group = task_to_group.get(to_task)
            if to_group is None or to_group == from_group:
                continue
            weights[(from_group, to_group)] += 1
    return dict(weights)


def group_level_stats(group_nodes, members_by_group, levels):
    stats = {}
    for group in group_nodes:
        group_id = group.get("id")
        group_levels = [levels.get(node.get("id"), 0) for node in members_by_group.get(group_id, [])]
        if group_levels:
            stats[group_id] = {
                "min_level": min(group_levels),
                "max_level": max(group_levels),
                "avg_level": sum(group_levels) / len(group_levels),
            }
        else:
            stats[group_id] = {
                "min_level": 0,
                "max_level": 0,
                "avg_level": 0,
            }
    return stats


def group_order_graph_cost(order_ids, weights, stats):
    positions = {group_id: index for index, group_id in enumerate(order_ids)}
    backward_penalty = 0
    span_penalty = 0

    for (from_group, to_group), weight in weights.items():
        distance = positions[to_group] - positions[from_group]
        span_penalty += weight * abs(distance)
        if distance < 0:
            backward_penalty += weight * abs(distance)

    return (
        backward_penalty,
        span_penalty,
        sum(stats[group_id]["min_level"] * (positions[group_id] + 1) for group_id in order_ids),
        tuple(order_ids),
    )


def exact_group_order(group_nodes, stats, weights):
    group_ids = [group.get("id") for group in group_nodes]
    if len(group_ids) > 7:
        return None

    best_order = None
    best_cost = None
    for permutation in itertools.permutations(group_ids):
        cost = group_order_graph_cost(permutation, weights, stats)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_order = permutation

    return list(best_order)


def order_groups(group_nodes, members_by_group, levels, outgoing):
    stats = group_level_stats(group_nodes, members_by_group, levels)
    default_order = sorted(
        group_nodes,
        key=lambda group: (
            stats[group.get("id")]["min_level"],
            stats[group.get("id")]["avg_level"],
            *stable_group_key(group),
        ),
    )
    weights = build_group_dependency_weights(default_order, members_by_group, outgoing)
    exact_order_ids = exact_group_order(default_order, stats, weights)
    if exact_order_ids is None:
        return default_order

    groups_by_id = {group.get("id"): group for group in default_order}
    return [groups_by_id[group_id] for group_id in exact_order_ids]


def group_base_ys(group_nodes, members_by_group, levels, base_y, config):
    stats = group_level_stats(group_nodes, members_by_group, levels)
    ranked_groups = sorted(
        group_nodes,
        key=lambda group: (
            stats[group.get("id")]["max_level"],
            stats[group.get("id")]["avg_level"],
            *stable_group_key(group),
        ),
    )
    positions = {}
    for rank, group in enumerate(ranked_groups):
        positions[group.get("id")] = base_y + (rank * config.group_band_step)
    return positions


def next_group_x(group, config):
    width = int(group.get("width", config.min_group_width))
    extra_gap = max(0, int(round((width - config.min_group_width) * config.group_width_gap_ratio)))
    return width + config.group_gap + extra_gap


def apply_group_arrangement(ordered_groups, band_order_ids, group_layouts, sizes, start_x, base_y, config):
    band_positions = {
        group_id: base_y + (rank * config.group_band_step)
        for rank, group_id in enumerate(band_order_ids)
    }
    x_cursor = start_x
    for group in ordered_groups:
        group_id = group.get("id")
        group["x"] = x_cursor
        group["y"] = band_positions[group_id]
        apply_group_layout(group, group_layouts[group_id], sizes, config)
        x_cursor += next_group_x(group, config)


def reflow_group_positions(ordered_groups, group_layouts, sizes, start_x, config):
    x_cursor = start_x
    for group in ordered_groups:
        group["x"] = x_cursor
        apply_group_layout(group, group_layouts[group.get("id")], sizes, config)
        x_cursor += next_group_x(group, config)


def reinsert_candidates(order_ids):
    for source_index in range(len(order_ids)):
        removed = order_ids[source_index]
        remainder = order_ids[:source_index] + order_ids[source_index + 1:]
        for target_index in range(len(order_ids)):
            candidate = remainder[:target_index] + [removed] + remainder[target_index:]
            if candidate != order_ids:
                yield candidate


def optimize_group_arrangement(canvas, ordered_groups, group_layouts, members_by_group, sizes, task_ids, start_x, base_y, config):
    order_ids = [group.get("id") for group in ordered_groups]
    band_order_ids = [
        group.get("id")
        for group in sorted(ordered_groups, key=lambda group: (group.get("y", 0), stable_group_key(group)))
    ]
    apply_group_arrangement(ordered_groups, band_order_ids, group_layouts, sizes, start_x, base_y, config)
    current_score = total_layout_score(canvas, task_ids, config)
    current_key = layout_score_order(current_score)

    for _ in range(3):
        changed = False

        for candidate_order_ids in reinsert_candidates(order_ids):
            candidate_groups = [next(group for group in ordered_groups if group.get("id") == group_id) for group_id in candidate_order_ids]
            apply_group_arrangement(candidate_groups, band_order_ids, group_layouts, sizes, start_x, base_y, config)
            score = total_layout_score(canvas, task_ids, config)
            if layout_score_order(score) < current_key:
                order_ids = candidate_order_ids
                ordered_groups[:] = candidate_groups
                current_score = score
                current_key = layout_score_order(score)
                changed = True
                break
        if changed:
            continue

        for candidate_band_ids in reinsert_candidates(band_order_ids):
            apply_group_arrangement(ordered_groups, candidate_band_ids, group_layouts, sizes, start_x, base_y, config)
            score = total_layout_score(canvas, task_ids, config)
            if layout_score_order(score) < current_key:
                band_order_ids = candidate_band_ids
                current_score = score
                current_key = layout_score_order(score)
                changed = True
                break

        if not changed:
            break

    apply_group_arrangement(ordered_groups, band_order_ids, group_layouts, sizes, start_x, base_y, config)
    return current_score


def build_row_positions(level_rows):
    positions = {}
    for level, row in level_rows.items():
        for index, node in enumerate(row):
            positions[node.get("id")] = (level, index)
    return positions


def reorder_level_by_barycenter(level, row, positions, same_group_ids, incoming, outgoing):
    scored = []
    for fallback_index, node in enumerate(row):
        node_id = node.get("id")
        neighbor_positions = []
        for neighbor_id in incoming.get(node_id, set()) | outgoing.get(node_id, set()):
            if neighbor_id not in same_group_ids:
                continue
            neighbor_level, neighbor_index = positions.get(neighbor_id, (None, None))
            if neighbor_level is None or neighbor_level == level:
                continue
            distance = abs(neighbor_level - level)
            weight = 1 / max(1, distance)
            neighbor_positions.append((neighbor_index, weight))

        if neighbor_positions:
            total_weight = sum(weight for _, weight in neighbor_positions)
            barycenter = sum(index * weight for index, weight in neighbor_positions) / total_weight
            has_neighbors = 0
        else:
            barycenter = fallback_index
            has_neighbors = 1

        scored.append(
            (
                has_neighbors,
                barycenter,
                fallback_index,
                *stable_task_key(node),
                node,
            )
        )

    return [item[-1] for item in sorted(scored)]


def order_tasks(group_tasks, levels, incoming, outgoing, config):
    tasks_by_level = defaultdict(list)
    same_group_ids = {node.get("id") for node in group_tasks}

    for node in group_tasks:
        tasks_by_level[levels.get(node.get("id"), 0)].append(node)

    ordered_rows = {
        level: sorted(nodes, key=stable_task_key)
        for level, nodes in tasks_by_level.items()
    }
    sorted_levels = sorted(ordered_rows)

    for _ in range(3):
        positions = build_row_positions(ordered_rows)
        for level in sorted_levels:
            ordered_rows[level] = reorder_level_by_barycenter(
                level,
                ordered_rows[level],
                positions,
                same_group_ids,
                incoming,
                outgoing,
            )
            positions = build_row_positions(ordered_rows)

        positions = build_row_positions(ordered_rows)
        for level in reversed(sorted_levels):
            ordered_rows[level] = reorder_level_by_barycenter(
                level,
                ordered_rows[level],
                positions,
                same_group_ids,
                incoming,
                outgoing,
            )
            positions = build_row_positions(ordered_rows)

    rows = []
    for level in sorted_levels:
        ordered_level = ordered_rows[level]
        for start in range(0, len(ordered_level), config.max_columns_per_level):
            stop = start + config.max_columns_per_level
            rows.append(
                {
                    "level": level,
                    "tasks": ordered_level[start:stop],
                    "last_in_level": stop >= len(ordered_level),
                }
            )
    return rows


def split_row_if_needed(row):
    tasks = row["tasks"]
    if len(tasks) <= 1:
        return [row]

    split_rows = []
    for index, node in enumerate(tasks):
        split_rows.append(
            {
                "level": row["level"],
                "tasks": [node],
                "last_in_level": row["last_in_level"] and index == len(tasks) - 1,
            }
        )
    return split_rows


def refine_rows_for_cross_group_visibility(ordered_groups, rows_by_group, outgoing):
    group_rank = {group.get("id"): index for index, group in enumerate(ordered_groups)}

    for _ in range(3):
        task_group = {}
        task_row = {}
        task_col = {}

        for group in ordered_groups:
            group_id = group.get("id")
            for row_index, row in enumerate(rows_by_group[group_id]):
                for col_index, node in enumerate(row["tasks"]):
                    task_id = node.get("id")
                    task_group[task_id] = group_id
                    task_row[task_id] = row_index
                    task_col[task_id] = col_index

        rows_to_split = set()
        for from_task, children in outgoing.items():
            from_group = task_group.get(from_task)
            if from_group is None:
                continue
            from_rank = group_rank[from_group]
            from_row_index = task_row[from_task]
            from_col_index = task_col[from_task]
            from_row = rows_by_group[from_group][from_row_index]

            for to_task in children:
                to_group = task_group.get(to_task)
                if to_group is None or to_group == from_group:
                    continue

                to_rank = group_rank[to_group]
                to_row_index = task_row[to_task]
                to_col_index = task_col[to_task]
                to_row = rows_by_group[to_group][to_row_index]

                if from_row_index != to_row_index:
                    continue

                if to_rank > from_rank:
                    if from_col_index < len(from_row["tasks"]) - 1:
                        rows_to_split.add((from_group, from_row_index))
                    if to_col_index > 0:
                        rows_to_split.add((to_group, to_row_index))
                elif to_rank < from_rank:
                    if from_col_index > 0:
                        rows_to_split.add((from_group, from_row_index))
                    if to_col_index < len(to_row["tasks"]) - 1:
                        rows_to_split.add((to_group, to_row_index))

        if not rows_to_split:
            break

        refined = {}
        for group in ordered_groups:
            group_id = group.get("id")
            new_rows = []
            for row_index, row in enumerate(rows_by_group[group_id]):
                if (group_id, row_index) in rows_to_split:
                    new_rows.extend(split_row_if_needed(row))
                else:
                    new_rows.append(row)
            refined[group_id] = new_rows
        rows_by_group = refined

    return rows_by_group


def point_in_rect(x, y, rect):
    left, top, width, height = rect
    return left <= x <= (left + width) and top <= y <= (top + height)


def rect_to_node(rect):
    left, top, width, height = rect
    return {"x": left, "y": top, "width": width, "height": height}


def node_center(node):
    return (
        node.get("x", 0) + (node.get("width", 0) / 2),
        node.get("y", 0) + (node.get("height", 0) / 2),
    )


def node_rect(node):
    return (
        node.get("x", 0),
        node.get("y", 0),
        node.get("width", 0),
        node.get("height", 0),
    )


def default_edge_side(from_node, to_node):
    from_center_x, from_center_y = node_center(from_node)
    to_center_x, to_center_y = node_center(to_node)
    angle = math.atan2(to_center_y - from_center_y, to_center_x - from_center_x)
    diagonal = math.atan2(from_node.get("height", 0), from_node.get("width", 0))

    if -diagonal < angle <= diagonal:
        return "right"
    if diagonal < angle <= math.pi - diagonal:
        return "bottom"
    if angle > math.pi - diagonal or angle <= -(math.pi - diagonal):
        return "left"
    return "top"


def face_midpoint(node, side):
    x = node.get("x", 0)
    y = node.get("y", 0)
    width = node.get("width", 0)
    height = node.get("height", 0)

    if side == "top":
        return (x + (width / 2), y)
    if side == "right":
        return (x + width, y + (height / 2))
    if side == "bottom":
        return (x + (width / 2), y + height)
    return (x, y + (height / 2))


def offset_point(side, point, distance):
    x, y = point
    if side == "top":
        return (x, y - distance)
    if side == "right":
        return (x + distance, y)
    if side == "bottom":
        return (x, y + distance)
    return (x - distance, y)


def orientation(ax, ay, bx, by, cx, cy):
    value = ((by - ay) * (cx - bx)) - ((bx - ax) * (cy - by))
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def on_segment(ax, ay, bx, by, cx, cy):
    return min(ax, cx) <= bx <= max(ax, cx) and min(ay, cy) <= by <= max(ay, cy)


def segments_intersect(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y):
    o1 = orientation(a1x, a1y, a2x, a2y, b1x, b1y)
    o2 = orientation(a1x, a1y, a2x, a2y, b2x, b2y)
    o3 = orientation(b1x, b1y, b2x, b2y, a1x, a1y)
    o4 = orientation(b1x, b1y, b2x, b2y, a2x, a2y)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(a1x, a1y, b1x, b1y, a2x, a2y):
        return True
    if o2 == 0 and on_segment(a1x, a1y, b2x, b2y, a2x, a2y):
        return True
    if o3 == 0 and on_segment(b1x, b1y, a1x, a1y, b2x, b2y):
        return True
    if o4 == 0 and on_segment(b1x, b1y, a2x, a2y, b2x, b2y):
        return True
    return False


def line_intersects_rect(x1, y1, x2, y2, rect, margin=0):
    left, top, width, height = rect
    left -= margin
    top -= margin
    width += 2 * margin
    height += 2 * margin
    right = left + width
    bottom = top + height

    if point_in_rect(x1, y1, (left, top, width, height)) or point_in_rect(x2, y2, (left, top, width, height)):
        return True

    edges = [
        (left, top, right, top),
        (right, top, right, bottom),
        (right, bottom, left, bottom),
        (left, bottom, left, top),
    ]
    return any(
        segments_intersect(x1, y1, x2, y2, ex1, ey1, ex2, ey2)
        for ex1, ey1, ex2, ey2 in edges
    )


def cubic_bezier_point(p0, p1, p2, p3, t):
    inv = 1 - t
    inv_sq = inv * inv
    t_sq = t * t
    x = (
        (inv_sq * inv * p0[0])
        + (3 * t * inv_sq * p1[0])
        + (3 * t_sq * inv * p2[0])
        + (t_sq * t * p3[0])
    )
    y = (
        (inv_sq * inv * p0[1])
        + (3 * t * inv_sq * p1[1])
        + (3 * t_sq * inv * p2[1])
        + (t_sq * t * p3[1])
    )
    return (x, y)


def obsidian_edge_points(from_node, from_side, to_node, to_side, config):
    start_face = face_midpoint(from_node, from_side)
    start_offset = offset_point(from_side, start_face, config.edge_port_gap)
    end_face = face_midpoint(to_node, to_side)
    end_offset = offset_point(to_side, end_face, config.edge_port_gap)

    distance = math.dist(start_offset, end_offset)
    handle = clamp(distance / 2, config.edge_handle_min, config.edge_handle_max)
    control_1 = offset_point(from_side, start_offset, handle)
    control_2 = offset_point(to_side, end_offset, handle)

    points = [start_face, start_offset]
    for step in range(1, config.edge_curve_samples):
        t = step / config.edge_curve_samples
        points.append(cubic_bezier_point(start_offset, control_1, control_2, end_offset, t))
    points.extend([end_offset, end_face])
    return points


def polyline_intersects_rect(points, rect, margin=0):
    poly_bounds = polyline_bounds(points)
    if not bounds_overlap(poly_bounds, rect_bounds(rect, margin=margin)):
        return False
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if line_intersects_rect(x1, y1, x2, y2, rect, margin=margin):
            return True
    return False


def polylines_intersect(points_a, points_b):
    if not bounds_overlap(polyline_bounds(points_a), polyline_bounds(points_b)):
        return False
    for (a1x, a1y), (a2x, a2y) in zip(points_a, points_a[1:]):
        for (b1x, b1y), (b2x, b2y) in zip(points_b, points_b[1:]):
            if segments_intersect(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y):
                return True
    return False


def iter_task_edges(canvas, task_ids):
    for edge in canvas.get("edges", []):
        from_id = edge.get("fromNode")
        to_id = edge.get("toNode")
        if from_id in task_ids and to_id in task_ids:
            yield edge


def edge_sides(edge, node_by_id):
    from_node = node_by_id[edge.get("fromNode")]
    to_node = node_by_id[edge.get("toNode")]
    preferred_from, preferred_to = SHARED.pick_sides(from_node, to_node)
    from_side = edge.get("fromSide")
    to_side = edge.get("toSide")
    if from_side not in SIDE_ORDER:
        from_side = preferred_from
    if to_side not in SIDE_ORDER:
        to_side = preferred_to
    return from_node, to_node, from_side, to_side


def edge_points(edge, node_by_id, config):
    from_node, to_node, from_side, to_side = edge_sides(edge, node_by_id)
    return obsidian_edge_points(from_node, from_side, to_node, to_side, config)


def polyline_bounds(points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def rect_bounds(rect, margin=0):
    left, top, width, height = rect
    return (left - margin, top - margin, left + width + margin, top + height + margin)


def bounds_overlap(left_bounds, right_bounds):
    return not (
        left_bounds[2] < right_bounds[0]
        or right_bounds[2] < left_bounds[0]
        or left_bounds[3] < right_bounds[1]
        or right_bounds[3] < left_bounds[1]
    )


def side_turn_distance(left, right):
    left_index = SIDE_INDEX[left]
    right_index = SIDE_INDEX[right]
    delta = abs(left_index - right_index)
    return min(delta, 4 - delta)


def side_alignment(side, dx, dy):
    magnitude = math.hypot(dx, dy)
    if magnitude <= 1e-9:
        return 1.0
    normal_x, normal_y = SIDE_NORMALS[side]
    return ((normal_x * dx) + (normal_y * dy)) / magnitude


def side_coherence_penalty(side, dx, dy):
    alignment = side_alignment(side, dx, dy)
    if alignment <= -0.6:
        return 4
    if alignment < -0.15:
        return 2
    if alignment < 0.25:
        return 1
    return 0


def edge_direction_penalty(edge, node_by_id):
    from_node, to_node, from_side, to_side = edge_sides(edge, node_by_id)
    preferred_from, preferred_to = SHARED.pick_sides(from_node, to_node)
    return side_turn_distance(from_side, preferred_from) + side_turn_distance(to_side, preferred_to)


def edge_coherence_penalty(edge, node_by_id):
    from_node, to_node, from_side, to_side = edge_sides(edge, node_by_id)
    from_center_x, from_center_y = node_center(from_node)
    to_center_x, to_center_y = node_center(to_node)
    dx = to_center_x - from_center_x
    dy = to_center_y - from_center_y
    return side_coherence_penalty(from_side, dx, dy) + side_coherence_penalty(to_side, -dx, -dy)


def edge_length(points):
    return sum(
        math.dist(start, end)
        for start, end in zip(points, points[1:])
    )


def total_edge_occlusion_penalty(canvas, task_ids, config):
    node_by_id = {
        node.get("id"): node
        for node in SHARED.get_tasks(canvas)
        if node.get("id") in task_ids
    }
    rects = {task_id: node_rect(node) for task_id, node in node_by_id.items()}
    penalty = 0

    for edge in canvas.get("edges", []):
        from_id = edge.get("fromNode")
        to_id = edge.get("toNode")
        if from_id not in node_by_id or to_id not in node_by_id:
            continue

        edge_points_value = edge_points(edge, node_by_id, config)

        for other_id, rect in rects.items():
            if other_id in {from_id, to_id}:
                continue
            if polyline_intersects_rect(edge_points_value, rect, margin=config.edge_occlusion_margin):
                penalty += 1

    return penalty


def total_edge_crossing_penalty(canvas, task_ids, config):
    node_by_id = {
        node.get("id"): node
        for node in SHARED.get_tasks(canvas)
        if node.get("id") in task_ids
    }
    edges = list(iter_task_edges(canvas, task_ids))
    edge_points_by_index = {
        index: edge_points(edge, node_by_id, config)
        for index, edge in enumerate(edges)
    }
    penalty = 0

    for left_index, left_edge in enumerate(edges):
        left_nodes = {left_edge.get("fromNode"), left_edge.get("toNode")}
        left_points = edge_points_by_index[left_index]
        for right_index in range(left_index + 1, len(edges)):
            right_edge = edges[right_index]
            right_nodes = {right_edge.get("fromNode"), right_edge.get("toNode")}
            if left_nodes & right_nodes:
                continue
            if polylines_intersect(left_points, edge_points_by_index[right_index]):
                penalty += 1

    return penalty


def total_edge_direction_penalty(canvas, task_ids):
    node_by_id = {
        node.get("id"): node
        for node in SHARED.get_tasks(canvas)
        if node.get("id") in task_ids
    }
    return sum(
        edge_direction_penalty(edge, node_by_id)
        for edge in iter_task_edges(canvas, task_ids)
    )


def total_edge_coherence_penalty(canvas, task_ids):
    node_by_id = {
        node.get("id"): node
        for node in SHARED.get_tasks(canvas)
        if node.get("id") in task_ids
    }
    return sum(
        edge_coherence_penalty(edge, node_by_id)
        for edge in iter_task_edges(canvas, task_ids)
    )


def side_flow_penalty(incoming_count, outgoing_count):
    return incoming_count * outgoing_count


def build_port_flow_counts(edges, node_by_id=None):
    counts = {}

    for edge in edges:
        from_id = edge.get("fromNode")
        to_id = edge.get("toNode")
        if node_by_id is None:
            from_side = edge.get("fromSide")
            to_side = edge.get("toSide")
        else:
            _, _, from_side, to_side = edge_sides(edge, node_by_id)

        if from_id and from_side in SIDE_ORDER:
            key = (from_id, from_side)
            incoming_count, outgoing_count = counts.get(key, (0, 0))
            counts[key] = (incoming_count, outgoing_count + 1)

        if to_id and to_side in SIDE_ORDER:
            key = (to_id, to_side)
            incoming_count, outgoing_count = counts.get(key, (0, 0))
            counts[key] = (incoming_count + 1, outgoing_count)

    return counts


def port_flow_penalty_from_counts(counts):
    return sum(
        side_flow_penalty(incoming_count, outgoing_count)
        for incoming_count, outgoing_count in counts.values()
    )


def total_port_flow_penalty(canvas, task_ids):
    node_by_id = {
        node.get("id"): node
        for node in SHARED.get_tasks(canvas)
        if node.get("id") in task_ids
    }
    counts = build_port_flow_counts(iter_task_edges(canvas, task_ids), node_by_id)
    return port_flow_penalty_from_counts(counts)


def total_edge_length_penalty(canvas, task_ids, config):
    node_by_id = {
        node.get("id"): node
        for node in SHARED.get_tasks(canvas)
        if node.get("id") in task_ids
    }
    return int(
        round(
            sum(
                edge_length(edge_points(edge, node_by_id, config))
                for edge in iter_task_edges(canvas, task_ids)
            )
        )
    )


def total_canvas_bbox_penalty(canvas, task_ids):
    nodes = [
        node
        for node in canvas.get("nodes", [])
        if node.get("id") in task_ids or node.get("type") == "group"
    ]
    if not nodes:
        return 0

    left = min(node.get("x", 0) for node in nodes)
    top = min(node.get("y", 0) for node in nodes)
    right = max(node.get("x", 0) + node.get("width", 0) for node in nodes)
    bottom = max(node.get("y", 0) + node.get("height", 0) for node in nodes)
    return int((right - left) * (bottom - top))


def total_layout_score(canvas, task_ids, config):
    return (
        total_edge_occlusion_penalty(canvas, task_ids, config),
        total_edge_coherence_penalty(canvas, task_ids),
        total_port_flow_penalty(canvas, task_ids),
        total_edge_crossing_penalty(canvas, task_ids, config),
        total_edge_direction_penalty(canvas, task_ids),
        total_canvas_bbox_penalty(canvas, task_ids),
        total_edge_length_penalty(canvas, task_ids, config),
    )


def layout_score_order(score):
    (
        occlusion,
        coherence_penalty,
        port_flow_penalty,
        crossings,
        direction_penalty,
        bbox_penalty,
        path_length,
    ) = score
    return (
        (occlusion * 5) + (coherence_penalty * 3) + (port_flow_penalty * 3) + crossings,
        occlusion,
        coherence_penalty,
        port_flow_penalty,
        crossings,
        direction_penalty,
        bbox_penalty,
        path_length,
    )


def group_rect(group, offset=0):
    return (
        group.get("x", 0),
        group.get("y", 0) + offset,
        group.get("width", 0),
        group.get("height", 0),
    )


def groups_overlap(ordered_groups, offsets, margin=60):
    rects = []
    for group in ordered_groups:
        rects.append(group_rect(group, offsets.get(group.get("id"), 0)))

    for index, left in enumerate(rects):
        for right in rects[index + 1:]:
            if not (
                left[0] + left[2] + margin <= right[0]
                or right[0] + right[2] + margin <= left[0]
                or left[1] + left[3] + margin <= right[1]
                or right[1] + right[3] + margin <= left[1]
            ):
                return True
    return False


def set_group_phase_offsets(ordered_groups, members_by_group, base_positions, offsets):
    for group in ordered_groups:
        group_id = group.get("id")
        base_y = base_positions[group_id]
        offset = offsets.get(group_id, 0)
        target_y = base_y + offset
        delta = target_y - group.get("y", 0)
        if delta == 0:
            continue
        group["y"] = target_y
        for node in members_by_group.get(group_id, []):
            node["y"] = node.get("y", 0) + delta


def optimize_group_phase_offsets(canvas, ordered_groups, members_by_group, task_ids, config):
    base_positions = {group.get("id"): group.get("y", 0) for group in ordered_groups}
    offsets = {group.get("id"): 0 for group in ordered_groups}
    candidate_offsets = [
        config.group_phase_step * step
        for step in range(-config.max_group_phase_steps, config.max_group_phase_steps + 1)
    ]
    current_score = total_layout_score(canvas, task_ids, config)

    for _ in range(4):
        changed = False
        for group in ordered_groups:
            group_id = group.get("id")
            current_offset = offsets[group_id]
            best_choice = (layout_score_order(current_score), current_score, abs(current_offset), 0, current_offset)

            for candidate in candidate_offsets:
                if candidate == current_offset:
                    continue
                trial_offsets = dict(offsets)
                trial_offsets[group_id] = candidate
                if groups_overlap(ordered_groups, trial_offsets):
                    continue
                set_group_phase_offsets(ordered_groups, members_by_group, base_positions, trial_offsets)
                score = total_layout_score(canvas, task_ids, config)
                choice = (
                    layout_score_order(score),
                    score,
                    abs(candidate),
                    abs(candidate - current_offset),
                    candidate,
                )
                if choice < best_choice:
                    best_choice = choice

            if best_choice[-1] != current_offset:
                offsets[group_id] = best_choice[-1]
                set_group_phase_offsets(ordered_groups, members_by_group, base_positions, offsets)
                current_score = best_choice[1]
                changed = True
            else:
                set_group_phase_offsets(ordered_groups, members_by_group, base_positions, offsets)

        if not changed:
            break

    set_group_phase_offsets(ordered_groups, members_by_group, base_positions, offsets)
    return offsets


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def compute_row_layout(rows, row_widths, row_heights, sizes, left_offsets, config):
    centers = {}
    rects = {}
    row_by_task = {}
    y_cursor = 0

    for row_index, row in enumerate(rows):
        row_left = left_offsets[row_index]
        cursor = row_left
        for node_index, node in enumerate(row["tasks"]):
            task_id = node.get("id")
            width, height = sizes[task_id]
            if node_index:
                cursor += config.task_gap
            rects[task_id] = (cursor, y_cursor, width, height)
            centers[task_id] = (cursor + (width / 2), y_cursor + (height / 2))
            row_by_task[task_id] = row_index
            cursor += width
        y_cursor += row_heights[row_index] + config.task_gap
        if row["last_in_level"]:
            y_cursor += config.level_gap

    return centers, rects, row_by_task


def long_edge_visibility_penalty(rows, row_widths, row_heights, sizes, left_offsets, outgoing, config):
    penalty = 0
    _, rects, row_by_task = compute_row_layout(
        rows,
        row_widths,
        row_heights,
        sizes,
        left_offsets,
        config,
    )

    for from_task, children in outgoing.items():
        from_row = row_by_task.get(from_task)
        if from_row is None:
            continue
        from_node = rect_to_node(rects[from_task])

        for to_task in children:
            to_row = row_by_task.get(to_task)
            if to_row is None:
                continue
            row_distance = abs(to_row - from_row)
            if row_distance <= 1:
                continue

            lower = min(from_row, to_row)
            upper = max(from_row, to_row)
            to_node = rect_to_node(rects[to_task])
            from_side = default_edge_side(from_node, to_node)
            to_side = default_edge_side(to_node, from_node)
            edge_points = obsidian_edge_points(
                from_node,
                from_side,
                to_node,
                to_side,
                config,
            )

            for task_id, rect in rects.items():
                if task_id in {from_task, to_task}:
                    continue
                row_index = row_by_task[task_id]
                if lower < row_index < upper:
                    if polyline_intersects_rect(edge_points, rect, margin=config.edge_occlusion_margin):
                        penalty += 1

    return penalty


def has_long_intra_group_edges(rows, outgoing):
    row_by_task = {}
    for row_index, row in enumerate(rows):
        for node in row["tasks"]:
            row_by_task[node.get("id")] = row_index

    for from_task, children in outgoing.items():
        from_row = row_by_task.get(from_task)
        if from_row is None:
            continue
        for to_task in children:
            to_row = row_by_task.get(to_task)
            if to_row is None:
                continue
            if abs(to_row - from_row) > 1:
                return True
    return False


def choose_visibility_aware_lanes(rows, row_widths, row_heights, sizes, left_offsets, content_width, outgoing, config):
    max_lefts = [max(0, content_width - row_width) for row_width in row_widths]

    for _ in range(2):
        for row_index, row in enumerate(rows):
            if len(row["tasks"]) != 1 or max_lefts[row_index] == 0:
                continue

            current_left = left_offsets[row_index]
            candidate_lefts = {
                0,
                int(round(max_lefts[row_index] / 2)),
                max_lefts[row_index],
                current_left,
            }

            best_choice = None
            for candidate in sorted(candidate_lefts):
                trial_lefts = list(left_offsets)
                trial_lefts[row_index] = candidate
                visibility_penalty = long_edge_visibility_penalty(
                    rows,
                    row_widths,
                    row_heights,
                    sizes,
                    trial_lefts,
                    outgoing,
                    config,
                )
                stability_penalty = abs(candidate - current_left) / 24
                centeredness_penalty = abs(candidate - int(round(max_lefts[row_index] / 2))) / 80
                choice = (visibility_penalty, stability_penalty, centeredness_penalty, candidate)
                if best_choice is None or choice < best_choice:
                    best_choice = choice

            left_offsets[row_index] = best_choice[-1]

    return left_offsets


def build_row_left_offsets(rows, row_widths, row_heights, sizes, content_width, incoming, outgoing, config):
    left_offsets = [int((content_width - row_width) / 2) for row_width in row_widths]

    def task_centers():
        centers, _, row_by_task = compute_row_layout(
            rows,
            row_widths,
            row_heights,
            sizes,
            left_offsets,
            config,
        )
        return centers, row_by_task

    for _ in range(3):
        centers, row_by_task = task_centers()
        for row_index, row in enumerate(rows):
            neighbor_centers = []
            for node in row["tasks"]:
                task_id = node.get("id")
                for neighbor_id in incoming.get(task_id, set()) | outgoing.get(task_id, set()):
                    neighbor_row = row_by_task.get(neighbor_id)
                    if neighbor_row is None or neighbor_row >= row_index:
                        continue
                    neighbor_centers.append(centers[neighbor_id][0])

            if neighbor_centers:
                target_center = sum(neighbor_centers) / len(neighbor_centers)
                target_left = int(round(target_center - (row_widths[row_index] / 2)))
                left_offsets[row_index] = clamp(
                    int(round((left_offsets[row_index] + target_left) / 2)),
                    0,
                    max(0, content_width - row_widths[row_index]),
                )
                centers, row_by_task = task_centers()

        centers, row_by_task = task_centers()
        for row_index in range(len(rows) - 1, -1, -1):
            row = rows[row_index]
            neighbor_centers = []
            for node in row["tasks"]:
                task_id = node.get("id")
                for neighbor_id in incoming.get(task_id, set()) | outgoing.get(task_id, set()):
                    neighbor_row = row_by_task.get(neighbor_id)
                    if neighbor_row is None or neighbor_row <= row_index:
                        continue
                    neighbor_centers.append(centers[neighbor_id][0])

            if neighbor_centers:
                target_center = sum(neighbor_centers) / len(neighbor_centers)
                target_left = int(round(target_center - (row_widths[row_index] / 2)))
                left_offsets[row_index] = clamp(
                    int(round((left_offsets[row_index] + target_left) / 2)),
                    0,
                    max(0, content_width - row_widths[row_index]),
                )
                centers, row_by_task = task_centers()

    left_offsets = choose_visibility_aware_lanes(
        rows,
        row_widths,
        row_heights,
        sizes,
        left_offsets,
        content_width,
        outgoing,
        config,
    )

    return left_offsets


def apply_group_layout(group, layout, sizes, config):
    group["width"] = layout["group_width"]
    group["height"] = layout["group_height"]

    y_cursor = group.get("y", 0) + config.group_padding_top
    for row_index, row in enumerate(layout["rows"]):
        row_x = group.get("x", 0) + config.group_padding_x + layout["row_lefts"][row_index]
        row_height = layout["row_heights"][row_index]

        for node in row["tasks"]:
            task_id = node.get("id")
            width, height = sizes[task_id]
            node["x"] = row_x
            node["y"] = y_cursor
            node["width"] = width
            node["height"] = height
            row_x += width + config.task_gap

        y_cursor += row_height + config.task_gap
        if row["last_in_level"]:
            y_cursor += config.level_gap


def normalize_last_in_level_flags(rows):
    for index, row in enumerate(rows):
        next_level = rows[index + 1]["level"] if index + 1 < len(rows) else None
        row["last_in_level"] = next_level != row["level"]


def level_row_spans(rows):
    spans = []
    start = 0
    while start < len(rows):
        level = rows[start]["level"]
        end = start + 1
        while end < len(rows) and rows[end]["level"] == level:
            end += 1
        spans.append((level, start, end))
        start = end
    return spans


def apply_block_permutation(sequence, start, end, permutation):
    original = list(sequence[start:end])
    reordered = [original[index] for index in permutation]
    sequence[start:end] = reordered


def set_block(sequence, start, end, values):
    sequence[start:end] = list(values)


def optimize_same_level_row_order(canvas, ordered_groups, group_layouts, sizes, task_ids, config):
    current_score = total_layout_score(canvas, task_ids, config)
    current_key = layout_score_order(current_score)

    for _ in range(3):
        changed = False
        for group in ordered_groups:
            group_id = group.get("id")
            layout = group_layouts[group_id]

            for _, start, end in level_row_spans(layout["rows"]):
                block_size = end - start
                if block_size <= 1:
                    continue

                current_permutation = tuple(range(block_size))
                if block_size > 5:
                    candidate_permutations = [
                        current_permutation,
                        tuple(reversed(current_permutation)),
                    ]
                else:
                    candidate_permutations = list(itertools.permutations(range(block_size)))

                original_rows = list(layout["rows"][start:end])
                original_widths = list(layout["row_widths"][start:end])
                original_heights = list(layout["row_heights"][start:end])
                original_lefts = list(layout["row_lefts"][start:end])

                best_choice = None
                for permutation in candidate_permutations:
                    if permutation == current_permutation:
                        continue

                    apply_block_permutation(layout["rows"], start, end, permutation)
                    apply_block_permutation(layout["row_widths"], start, end, permutation)
                    apply_block_permutation(layout["row_heights"], start, end, permutation)
                    apply_block_permutation(layout["row_lefts"], start, end, permutation)
                    normalize_last_in_level_flags(layout["rows"])
                    apply_group_layout(group, layout, sizes, config)

                    score = total_layout_score(canvas, task_ids, config)
                    moved_rows = sum(1 for index, value in enumerate(permutation) if index != value)
                    choice = (layout_score_order(score), score, moved_rows, permutation)
                    if best_choice is None or choice < best_choice:
                        best_choice = choice

                    set_block(layout["rows"], start, end, original_rows)
                    set_block(layout["row_widths"], start, end, original_widths)
                    set_block(layout["row_heights"], start, end, original_heights)
                    set_block(layout["row_lefts"], start, end, original_lefts)
                    normalize_last_in_level_flags(layout["rows"])
                    apply_group_layout(group, layout, sizes, config)

                if best_choice is None:
                    continue

                _, best_score, _, best_permutation = best_choice
                if layout_score_order(best_score) < current_key:
                    apply_block_permutation(layout["rows"], start, end, best_permutation)
                    apply_block_permutation(layout["row_widths"], start, end, best_permutation)
                    apply_block_permutation(layout["row_heights"], start, end, best_permutation)
                    apply_block_permutation(layout["row_lefts"], start, end, best_permutation)
                    normalize_last_in_level_flags(layout["rows"])
                    apply_group_layout(group, layout, sizes, config)
                    current_score = best_score
                    current_key = layout_score_order(best_score)
                    changed = True

        if not changed:
            break

    return current_score


def build_row_left_candidates(current_left, max_left, config):
    centered = int(round(max_left / 2))
    quarter = int(round(max_left / 4))
    three_quarters = int(round((3 * max_left) / 4))
    step = max(config.task_gap, int(config.long_edge_lane_padding / 2))
    candidates = {0, quarter, centered, three_quarters, max_left, current_left}

    for candidate in range(0, max_left + step, step):
        candidates.add(clamp(candidate, 0, max_left))

    for delta in (-2 * step, -step, step, 2 * step):
        candidates.add(clamp(current_left + delta, 0, max_left))

    return sorted(candidates)


def optimize_group_row_left_offsets(canvas, group, layout, sizes, task_ids, config, current_score=None):
    if current_score is None:
        current_score = total_layout_score(canvas, task_ids, config)
    current_key = layout_score_order(current_score)
    changed = False

    for row_index, row_width in enumerate(layout["row_widths"]):
        max_left = max(0, layout["content_width"] - row_width)
        if max_left == 0:
            continue

        current_left = layout["row_lefts"][row_index]
        centered = int(round(max_left / 2))
        best_choice = None

        for candidate in build_row_left_candidates(current_left, max_left, config):
            if candidate == current_left:
                continue

            layout["row_lefts"][row_index] = candidate
            apply_group_layout(group, layout, sizes, config)
            score = total_layout_score(canvas, task_ids, config)
            choice = (
                layout_score_order(score),
                score,
                abs(candidate - centered),
                abs(candidate - current_left),
                candidate,
            )
            if best_choice is None or choice < best_choice:
                best_choice = choice

        layout["row_lefts"][row_index] = current_left
        apply_group_layout(group, layout, sizes, config)

        if best_choice is None:
            continue

        _, best_score, _, _, best_left = best_choice
        if layout_score_order(best_score) < current_key:
            layout["row_lefts"][row_index] = best_left
            apply_group_layout(group, layout, sizes, config)
            current_score = best_score
            current_key = layout_score_order(best_score)
            changed = True

    return current_score, changed


def optimize_row_left_offsets(canvas, ordered_groups, group_layouts, sizes, task_ids, config):
    current_score = total_layout_score(canvas, task_ids, config)

    for _ in range(4):
        changed = False
        for group in ordered_groups:
            group_id = group.get("id")
            layout = group_layouts[group_id]
            current_score, group_changed = optimize_group_row_left_offsets(
                canvas,
                group,
                layout,
                sizes,
                task_ids,
                config,
                current_score=current_score,
            )
            changed = changed or group_changed

        if not changed:
            break

    return current_score


def candidate_group_content_widths(layout, config):
    base = layout["min_content_width"]
    candidates = {
        base,
        max(base, layout["max_row_width"]),
        max(base, layout["max_row_width"] + int(config.long_edge_lane_padding / 2)),
        max(base, layout["max_row_width"] + config.long_edge_lane_padding),
        layout["content_width"],
    }
    if layout["has_long_edges"]:
        for multiplier in (2, 3, 4):
            candidates.add(max(base, layout["max_row_width"] + (multiplier * config.long_edge_lane_padding)))
    return sorted(candidates)


def set_group_content_width(group, layout, content_width, sizes, config):
    layout["content_width"] = max(layout["min_content_width"], int(content_width))
    layout["group_width"] = max(config.min_group_width, layout["content_width"] + (2 * config.group_padding_x))
    max_lefts = [max(0, layout["content_width"] - row_width) for row_width in layout["row_widths"]]
    current_lefts = list(layout["row_lefts"])
    resized_lefts = []
    for current_left, previous_max_left, new_max_left in zip(current_lefts, layout["max_lefts"], max_lefts):
        if previous_max_left <= 0:
            resized_lefts.append(0)
            continue
        ratio = current_left / previous_max_left
        resized_lefts.append(clamp(int(round(ratio * new_max_left)), 0, new_max_left))
    layout["row_lefts"] = resized_lefts
    layout["max_lefts"] = max_lefts
    apply_group_layout(group, layout, sizes, config)


def optimize_group_content_widths(canvas, ordered_groups, group_layouts, sizes, task_ids, start_x, config):
    current_score = total_layout_score(canvas, task_ids, config)
    current_key = layout_score_order(current_score)

    for _ in range(3):
        changed = False
        for group in ordered_groups:
            group_id = group.get("id")
            layout = group_layouts[group_id]
            current_width = layout["content_width"]
            current_lefts = list(layout["row_lefts"])
            best_choice = None

            for candidate_width in candidate_group_content_widths(layout, config):
                if candidate_width == current_width:
                    continue
                set_group_content_width(group, layout, candidate_width, sizes, config)
                reflow_group_positions(ordered_groups, group_layouts, sizes, start_x, config)
                score, _ = optimize_group_row_left_offsets(
                    canvas,
                    group,
                    layout,
                    sizes,
                    task_ids,
                    config,
                )
                choice = (
                    layout_score_order(score),
                    score,
                    abs(candidate_width - current_width),
                    tuple(layout["row_lefts"]),
                    candidate_width,
                )
                if best_choice is None or choice < best_choice:
                    best_choice = choice
                layout["row_lefts"] = list(current_lefts)
                apply_group_layout(group, layout, sizes, config)
                reflow_group_positions(ordered_groups, group_layouts, sizes, start_x, config)

            set_group_content_width(group, layout, current_width, sizes, config)
            layout["row_lefts"] = list(current_lefts)
            apply_group_layout(group, layout, sizes, config)
            reflow_group_positions(ordered_groups, group_layouts, sizes, start_x, config)
            if best_choice is None:
                continue

            _, best_score, _, best_lefts, best_width = best_choice
            if layout_score_order(best_score) < current_key:
                set_group_content_width(group, layout, best_width, sizes, config)
                layout["row_lefts"] = list(best_lefts)
                apply_group_layout(group, layout, sizes, config)
                reflow_group_positions(ordered_groups, group_layouts, sizes, start_x, config)
                current_score = best_score
                current_key = layout_score_order(best_score)
                changed = True

        if not changed:
            break

    return current_score


def layout_group_contents(group, rows, sizes, incoming, outgoing, x_cursor, base_y, config):
    group["x"] = x_cursor
    group["y"] = base_y

    content_width = 0
    row_specs = []
    for row in rows:
        row_width = 0
        row_height = 0
        for index, node in enumerate(row["tasks"]):
            width, height = sizes[node.get("id")]
            row_width += width
            if index:
                row_width += config.task_gap
            row_height = max(row_height, height)
        row_specs.append((row, row_width, row_height))
        content_width = max(content_width, row_width)

    row_widths = [row_width for _, row_width, _ in row_specs]
    max_row_width = max(row_widths, default=0)
    min_content_width = max(content_width, config.min_group_width - (2 * config.group_padding_x))
    content_width = min_content_width
    group["width"] = max(config.min_group_width, content_width + (2 * config.group_padding_x))
    row_lefts = build_row_left_offsets(
        [row for row, _, _ in row_specs],
        [row_width for _, row_width, _ in row_specs],
        [row_height for _, _, row_height in row_specs],
        sizes,
        content_width,
        incoming,
        outgoing,
        config,
    )

    y_cursor = base_y + config.group_padding_top
    content_bottom = y_cursor

    for row_index, (row, row_width, row_height) in enumerate(row_specs):
        row_x = x_cursor + config.group_padding_x + row_lefts[row_index]
        for node in row["tasks"]:
            task_id = node.get("id")
            width, height = sizes[task_id]
            node["x"] = row_x
            node["y"] = y_cursor
            node["width"] = width
            node["height"] = height
            row_x += width + config.task_gap

        content_bottom = max(content_bottom, y_cursor + row_height)
        y_cursor += row_height + config.task_gap
        if row["last_in_level"]:
            y_cursor += config.level_gap

    if rows:
        group_height = max(
            config.min_group_height,
            (content_bottom - base_y) + config.group_padding_bottom,
        )
    else:
        group_height = max(config.min_group_height, int(group.get("height", config.min_group_height)))

    layout = {
        "rows": [row for row, _, _ in row_specs],
        "row_widths": [row_width for _, row_width, _ in row_specs],
        "row_heights": [row_height for _, _, row_height in row_specs],
        "row_lefts": row_lefts,
        "content_width": content_width,
        "min_content_width": min_content_width,
        "max_row_width": max_row_width,
        "has_long_edges": has_long_intra_group_edges(rows, outgoing),
        "max_lefts": [max(0, content_width - row_width) for row_width in row_widths],
        "group_width": group["width"],
        "group_height": group_height,
    }
    apply_group_layout(group, layout, sizes, config)
    return layout


def stable_edge_key(edge):
    return (
        edge.get("fromNode", ""),
        edge.get("toNode", ""),
        edge.get("id", ""),
    )


def compute_edge_score_data(edge, node_by_id, rects, config):
    points = edge_points(edge, node_by_id, config)
    bounds = polyline_bounds(points)
    from_id = edge.get("fromNode")
    to_id = edge.get("toNode")
    occlusion = 0

    for other_id, rect in rects.items():
        if other_id in {from_id, to_id}:
            continue
        if polyline_intersects_rect(points, rect, margin=config.edge_occlusion_margin):
            occlusion += 1

    return {
        "points": points,
        "bounds": bounds,
        "occlusion": occlusion,
        "coherence": edge_coherence_penalty(edge, node_by_id),
        "direction": edge_direction_penalty(edge, node_by_id),
        "length": int(round(edge_length(points))),
        "nodes": {from_id, to_id},
    }


def edge_pair_crosses(left_info, right_info):
    if left_info["nodes"] & right_info["nodes"]:
        return False
    if not bounds_overlap(left_info["bounds"], right_info["bounds"]):
        return False
    return polylines_intersect(left_info["points"], right_info["points"])


def optimize_edge_attachment_sides(canvas, task_ids, config):
    node_by_id = {node.get("id"): node for node in canvas.get("nodes", [])}
    rects = {
        task_id: node_rect(node)
        for task_id, node in node_by_id.items()
        if task_id in task_ids
    }
    task_edges = sorted(iter_task_edges(canvas, task_ids), key=stable_edge_key)

    for edge in task_edges:
        from_node = node_by_id[edge.get("fromNode")]
        to_node = node_by_id[edge.get("toNode")]
        preferred_from, preferred_to = SHARED.pick_sides(from_node, to_node)
        if edge.get("fromSide") not in SIDE_ORDER:
            edge["fromSide"] = preferred_from
        if edge.get("toSide") not in SIDE_ORDER:
            edge["toSide"] = preferred_to

    edge_infos = [
        compute_edge_score_data(edge, node_by_id, rects, config)
        for edge in task_edges
    ]
    crossing_neighbors = [set() for _ in task_edges]
    total_crossings = 0

    for left_index in range(len(task_edges)):
        left_info = edge_infos[left_index]
        for right_index in range(left_index + 1, len(task_edges)):
            if edge_pair_crosses(left_info, edge_infos[right_index]):
                crossing_neighbors[left_index].add(right_index)
                crossing_neighbors[right_index].add(left_index)
                total_crossings += 1

    total_occlusion = sum(info["occlusion"] for info in edge_infos)
    total_coherence = sum(info["coherence"] for info in edge_infos)
    port_flow_counts = build_port_flow_counts(task_edges)
    total_mixed_ports = port_flow_penalty_from_counts(port_flow_counts)
    total_direction = sum(info["direction"] for info in edge_infos)
    total_bbox = total_canvas_bbox_penalty(canvas, task_ids)
    total_length = sum(info["length"] for info in edge_infos)
    current_score = (
        total_occlusion,
        total_coherence,
        total_mixed_ports,
        total_crossings,
        total_direction,
        total_bbox,
        total_length,
    )
    current_key = layout_score_order(current_score)

    for _ in range(4):
        changed = False
        for edge_index, edge in enumerate(task_edges):
            current_from = edge["fromSide"]
            current_to = edge["toSide"]
            current_info = edge_infos[edge_index]
            current_neighbors = set(crossing_neighbors[edge_index])
            current_crossings = len(current_neighbors)
            best_choice = (
                current_key,
                current_score,
                current_info["coherence"],
                total_mixed_ports,
                current_info["direction"],
                current_info["length"],
                current_from,
                current_to,
                current_info,
                current_neighbors,
                {},
            )

            for from_side in SIDE_ORDER:
                for to_side in SIDE_ORDER:
                    if from_side == current_from and to_side == current_to:
                        continue
                    edge["fromSide"] = from_side
                    edge["toSide"] = to_side
                    candidate_info = compute_edge_score_data(edge, node_by_id, rects, config)
                    candidate_neighbors = set()
                    candidate_crossings = 0
                    for other_index, other_info in enumerate(edge_infos):
                        if other_index == edge_index:
                            continue
                        if edge_pair_crosses(candidate_info, other_info):
                            candidate_neighbors.add(other_index)
                            candidate_crossings += 1

                    current_from_key = (edge.get("fromNode"), current_from)
                    current_to_key = (edge.get("toNode"), current_to)
                    candidate_from_key = (edge.get("fromNode"), from_side)
                    candidate_to_key = (edge.get("toNode"), to_side)
                    affected_keys = {
                        current_from_key,
                        current_to_key,
                        candidate_from_key,
                        candidate_to_key,
                    }
                    current_mixed_penalty = sum(
                        side_flow_penalty(*port_flow_counts.get(key, (0, 0)))
                        for key in affected_keys
                    )
                    candidate_port_flow_counts = {
                        key: list(port_flow_counts.get(key, (0, 0)))
                        for key in affected_keys
                    }
                    candidate_port_flow_counts[current_from_key][1] -= 1
                    candidate_port_flow_counts[current_to_key][0] -= 1
                    candidate_port_flow_counts[candidate_from_key][1] += 1
                    candidate_port_flow_counts[candidate_to_key][0] += 1
                    candidate_mixed_penalty = sum(
                        side_flow_penalty(incoming_count, outgoing_count)
                        for incoming_count, outgoing_count in candidate_port_flow_counts.values()
                    )

                    score = (
                        total_occlusion - current_info["occlusion"] + candidate_info["occlusion"],
                        total_coherence - current_info["coherence"] + candidate_info["coherence"],
                        total_mixed_ports - current_mixed_penalty + candidate_mixed_penalty,
                        total_crossings - current_crossings + candidate_crossings,
                        total_direction - current_info["direction"] + candidate_info["direction"],
                        total_bbox,
                        total_length - current_info["length"] + candidate_info["length"],
                    )
                    choice = (
                        layout_score_order(score),
                        score,
                        candidate_info["coherence"],
                        score[2],
                        candidate_info["direction"],
                        candidate_info["length"],
                        from_side,
                        to_side,
                        candidate_info,
                        candidate_neighbors,
                        candidate_port_flow_counts,
                    )
                    if best_choice is None or choice < best_choice:
                        best_choice = choice

            edge["fromSide"] = current_from
            edge["toSide"] = current_to

            _, best_score, _, _, _, _, best_from, best_to, best_info, best_neighbors, best_port_flow_counts = best_choice
            if layout_score_order(best_score) < current_key:
                edge["fromSide"] = best_from
                edge["toSide"] = best_to
                removed_neighbors = current_neighbors - best_neighbors
                added_neighbors = best_neighbors - current_neighbors
                for other_index in removed_neighbors:
                    crossing_neighbors[other_index].discard(edge_index)
                for other_index in added_neighbors:
                    crossing_neighbors[other_index].add(edge_index)
                crossing_neighbors[edge_index] = best_neighbors
                edge_infos[edge_index] = best_info
                for key, counts in best_port_flow_counts.items():
                    port_flow_counts[key] = tuple(counts)
                (
                    total_occlusion,
                    total_coherence,
                    total_mixed_ports,
                    total_crossings,
                    total_direction,
                    total_bbox,
                    total_length,
                ) = best_score
                current_score = best_score
                current_key = layout_score_order(best_score)
                changed = True
            else:
                edge["fromSide"] = current_from
                edge["toSide"] = current_to

        if not changed:
            break

    return current_score


def organize_canvas(canvas, config):
    original_canvas = copy.deepcopy(canvas)
    updated = copy.deepcopy(canvas)
    task_nodes = SHARED.get_tasks(updated)
    group_nodes = SHARED.get_groups(updated)

    if not task_nodes or not group_nodes:
        raise RuntimeError("Canvas must contain workflow tasks and groups")

    incoming, outgoing = build_task_graph(updated, task_nodes)
    levels = longest_path_levels(task_nodes, incoming, outgoing)
    sizes = {node.get("id"): estimate_card_size(node, config) for node in task_nodes}
    members_by_group, ungrouped = infer_group_membership(updated, task_nodes)

    base_y = min((group.get("y", 0) for group in group_nodes), default=0)
    x_cursor = min((group.get("x", 0) for group in group_nodes), default=0)
    ordered_groups = order_groups(group_nodes, members_by_group, levels, outgoing)
    group_start_ys = group_base_ys(ordered_groups, members_by_group, levels, base_y, config)
    rows_by_group = {}
    group_layouts = {}

    for group in ordered_groups:
        rows_by_group[group.get("id")] = order_tasks(
            members_by_group.get(group.get("id"), []),
            levels,
            incoming,
            outgoing,
            config,
        )

    rows_by_group = refine_rows_for_cross_group_visibility(
        ordered_groups,
        rows_by_group,
        outgoing,
    )

    for group in ordered_groups:
        ordered_rows = rows_by_group[group.get("id")]
        group_layouts[group.get("id")] = layout_group_contents(
            group,
            ordered_rows,
            sizes,
            incoming,
            outgoing,
            x_cursor,
            group_start_ys[group.get("id")],
            config,
        )
        x_cursor += next_group_x(group, config)

    task_ids = {node.get("id") for node in task_nodes}
    optimize_same_level_row_order(updated, ordered_groups, group_layouts, sizes, task_ids, config)
    start_x = min((group.get("x", 0) for group in ordered_groups), default=0)
    optimize_group_content_widths(updated, ordered_groups, group_layouts, sizes, task_ids, start_x, config)
    optimize_row_left_offsets(updated, ordered_groups, group_layouts, sizes, task_ids, config)
    optimize_group_arrangement(
        updated,
        ordered_groups,
        group_layouts,
        members_by_group,
        sizes,
        task_ids,
        start_x,
        base_y,
        config,
    )
    baseline_canvas = copy.deepcopy(updated)
    optimize_edge_attachment_sides(baseline_canvas, task_ids, config)
    baseline_score = total_layout_score(baseline_canvas, task_ids, config)

    optimize_group_phase_offsets(
        updated,
        ordered_groups,
        members_by_group,
        task_ids,
        config,
    )
    optimize_edge_attachment_sides(updated, task_ids, config)
    for _ in range(2):
        previous_score = total_layout_score(updated, task_ids, config)
        optimize_same_level_row_order(updated, ordered_groups, group_layouts, sizes, task_ids, config)
        start_x = min((group.get("x", 0) for group in ordered_groups), default=0)
        optimize_group_content_widths(updated, ordered_groups, group_layouts, sizes, task_ids, start_x, config)
        optimize_row_left_offsets(updated, ordered_groups, group_layouts, sizes, task_ids, config)
        reflow_group_positions(ordered_groups, group_layouts, sizes, start_x, config)
        optimize_group_phase_offsets(
            updated,
            ordered_groups,
            members_by_group,
            task_ids,
            config,
        )
        optimize_edge_attachment_sides(updated, task_ids, config)
        candidate_score = total_layout_score(updated, task_ids, config)
        if not layout_score_order(candidate_score) < layout_score_order(previous_score):
            break
    score = total_layout_score(updated, task_ids, config)
    if layout_score_order(baseline_score) <= layout_score_order(score):
        updated = baseline_canvas
        score = baseline_score

    original_task_ids = {node.get("id") for node in SHARED.get_tasks(original_canvas)}
    original_score = total_layout_score(original_canvas, original_task_ids, config)
    if layout_score_order(original_score) <= layout_score_order(score):
        updated = original_canvas
        score = original_score

    final_score = total_layout_score(updated, task_ids, config)

    return updated, {
        "tasks": len(task_nodes),
        "groups": len(group_nodes),
        "ungrouped_tasks": len(ungrouped),
        "max_level": max(levels.values(), default=0),
        "occlusion_score": final_score[0],
        "coherence_penalty": final_score[1],
        "port_flow_penalty": final_score[2],
        "edge_crossings": final_score[3],
        "direction_penalty": final_score[4],
        "preserved_input": layout_score_order(original_score) <= layout_score_order(score),
    }


def default_output_path(canvas_file):
    path = Path(canvas_file)
    return path.with_name(f"{path.stem}.organized{path.suffix}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone sandbox CLI for experimental DAG-based canvas organization."
    )
    parser.add_argument("canvas_file", help="Path to the .canvas file to organize")
    parser.add_argument(
        "--output",
        help="Output path. Defaults to <input>.organized.canvas unless --in-place is used.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the input canvas in place.",
    )
    parser.add_argument("--group-gap", type=int, default=180, help="Horizontal gap between groups")
    parser.add_argument("--task-gap", type=int, default=40, help="Base gap between task cards")
    parser.add_argument("--level-gap", type=int, default=30, help="Extra vertical gap between dependency levels")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.in_place and args.output:
        parser.error("--output and --in-place cannot be used together")

    config = LayoutConfig(
        group_gap=args.group_gap,
        task_gap=args.task_gap,
        level_gap=args.level_gap,
    )

    canvas = SHARED.load_canvas(args.canvas_file)
    updated, summary = organize_canvas(canvas, config)

    output_path = Path(args.canvas_file) if args.in_place else Path(args.output or default_output_path(args.canvas_file))
    SHARED.save_canvas(str(output_path), updated)

    print(f"Organized {summary['tasks']} tasks across {summary['groups']} groups.")
    print(f"Ungrouped tasks left in place: {summary['ungrouped_tasks']}")
    print(f"Longest dependency level: {summary['max_level']}")
    print(f"Card occlusion score: {summary['occlusion_score']}")
    print(f"Port coherence penalty: {summary['coherence_penalty']}")
    print(f"Port flow penalty: {summary['port_flow_penalty']}")
    print(f"Edge crossing score: {summary['edge_crossings']}")
    print(f"Port direction penalty: {summary['direction_penalty']}")
    if summary.get("preserved_input"):
        print("Preserved original layout because it scored better than the generated layout.")
    print(f"Wrote organized canvas to: {output_path}")


if __name__ == "__main__":
    main()
