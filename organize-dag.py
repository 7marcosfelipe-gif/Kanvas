#!/usr/bin/env python3
"""Pure DAG canvas organizer without groups.

This tool ignores group constraints and lays out task cards with a layered
Sugiyama-style DAG pass:
1. Longest-path layering
2. Proper layered crossing reduction with dummy nodes and median sweeps
3. Coordinate assignment for real nodes only
4. Edge-side optimization using the existing organizer heuristics
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def load_base_module():
    module_path = Path(__file__).with_name("organize-canvas.py")
    spec = importlib.util.spec_from_file_location("organize_canvas_base", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared organizer helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


@dataclass(frozen=True)
class DagLayoutConfig:
    layer_gap: int = 260
    node_gap: int = 60
    left_margin: int = 80
    top_margin: int = 80
    sweep_passes: int = 8
    coord_iterations: int = 6
    drop_groups: bool = True


def stable_node_id(node_id, task_by_id):
    node = task_by_id.get(node_id)
    if node is None:
        return (node_id,)
    return BASE.stable_task_key(node)


def build_proper_layers(task_nodes, levels, outgoing):
    task_by_id = {node.get("id"): node for node in task_nodes}
    max_layer = max(levels.values(), default=0)
    layers = [[] for _ in range(max_layer + 1)]
    parents = defaultdict(set)
    children = defaultdict(set)
    dummy_ids = set()
    dummy_index = 0

    for node in sorted(task_nodes, key=BASE.stable_task_key):
        layers[levels[node.get("id")]].append(node.get("id"))

    for from_node in sorted(task_by_id, key=lambda item: stable_node_id(item, task_by_id)):
        for to_node in sorted(
            outgoing[from_node],
            key=lambda item: (levels[item], stable_node_id(item, task_by_id)),
        ):
            prev_node = from_node
            for layer in range(levels[from_node] + 1, levels[to_node]):
                dummy_id = f"__dummy_{dummy_index}"
                dummy_index += 1
                dummy_ids.add(dummy_id)
                layers[layer].append(dummy_id)
                children[prev_node].add(dummy_id)
                parents[dummy_id].add(prev_node)
                prev_node = dummy_id
            children[prev_node].add(to_node)
            parents[to_node].add(prev_node)

    return layers, parents, children, dummy_ids


def median_position(node_id, relation, adjacent_positions, fallback):
    values = sorted(
        adjacent_positions[neighbor]
        for neighbor in relation.get(node_id, ())
        if neighbor in adjacent_positions
    )
    if not values:
        return (1, fallback, fallback)
    return (0, statistics.median(values), fallback)


def order_layer(layer_nodes, relation, adjacent_positions):
    fallback_positions = {node_id: index for index, node_id in enumerate(layer_nodes)}
    decorated = [
        (
            median_position(node_id, relation, adjacent_positions, fallback_positions[node_id]),
            fallback_positions[node_id],
            node_id,
        )
        for node_id in layer_nodes
    ]
    return [node_id for _, _, node_id in sorted(decorated)]


def reduce_crossings(layers, parents, children, sweep_passes):
    ordered_layers = [list(layer) for layer in layers]

    for _ in range(max(1, sweep_passes)):
        for layer_index in range(1, len(ordered_layers)):
            adjacent_positions = {
                node_id: index
                for index, node_id in enumerate(ordered_layers[layer_index - 1])
            }
            ordered_layers[layer_index] = order_layer(
                ordered_layers[layer_index],
                parents,
                adjacent_positions,
            )

        for layer_index in range(len(ordered_layers) - 2, -1, -1):
            adjacent_positions = {
                node_id: index
                for index, node_id in enumerate(ordered_layers[layer_index + 1])
            }
            ordered_layers[layer_index] = order_layer(
                ordered_layers[layer_index],
                children,
                adjacent_positions,
            )

    return ordered_layers


def adjacent_layer_crossings(left_layer, right_layer, children):
    if not left_layer or not right_layer:
        return 0

    right_positions = {node_id: index for index, node_id in enumerate(right_layer)}
    edges = []
    for left_index, left_node in enumerate(left_layer):
        for right_node in children.get(left_node, ()):
            if right_node in right_positions:
                edges.append((left_index, right_positions[right_node]))

    crossings = 0
    for index, (left_a, right_a) in enumerate(edges):
        for left_b, right_b in edges[index + 1:]:
            if (left_a < left_b and right_a > right_b) or (left_a > left_b and right_a < right_b):
                crossings += 1
    return crossings


def local_layer_crossings(layers, children, layer_index):
    crossings = 0
    if layer_index > 0:
        crossings += adjacent_layer_crossings(layers[layer_index - 1], layers[layer_index], children)
    if layer_index + 1 < len(layers):
        crossings += adjacent_layer_crossings(layers[layer_index], layers[layer_index + 1], children)
    return crossings


def transpose_layers(layers, children):
    ordered_layers = [list(layer) for layer in layers]

    for _ in range(6):
        changed = False
        for layer_index, layer in enumerate(ordered_layers):
            if len(layer) < 2:
                continue

            made_local_change = True
            while made_local_change:
                made_local_change = False
                for index in range(len(layer) - 1):
                    before = local_layer_crossings(ordered_layers, children, layer_index)
                    layer[index], layer[index + 1] = layer[index + 1], layer[index]
                    after = local_layer_crossings(ordered_layers, children, layer_index)
                    if after < before:
                        changed = True
                        made_local_change = True
                    else:
                        layer[index], layer[index + 1] = layer[index + 1], layer[index]

        if not changed:
            break

    return ordered_layers


def extract_real_layers(layers, dummy_ids):
    return [
        [node_id for node_id in layer if node_id not in dummy_ids]
        for layer in layers
    ]


def median_center(node_id, relation, centers, fallback):
    values = sorted(
        centers[neighbor]
        for neighbor in relation.get(node_id, ())
        if neighbor in centers
    )
    if not values:
        return fallback
    return statistics.median(values)


def place_layer(layer_nodes, desired_centers, heights, gap):
    if not layer_nodes:
        return {}

    tops = {}
    cursor = None
    for node_id in layer_nodes:
        height = heights[node_id]
        target_top = desired_centers[node_id] - (height / 2)
        if cursor is None:
            top = target_top
        else:
            top = max(target_top, cursor + gap)
        tops[node_id] = top
        cursor = top + height

    actual_mean = statistics.mean(tops[node_id] + (heights[node_id] / 2) for node_id in layer_nodes)
    desired_mean = statistics.mean(desired_centers[node_id] for node_id in layer_nodes)
    shift = desired_mean - actual_mean
    if shift:
        for node_id in layer_nodes:
            tops[node_id] += shift

    cursor = None
    for node_id in layer_nodes:
        top = tops[node_id]
        if cursor is not None:
            top = max(top, cursor + gap)
        tops[node_id] = top
        cursor = top + heights[node_id]

    return tops


def assign_vertical_positions(real_layers, incoming, outgoing, sizes, config):
    heights = {node_id: size[1] for node_id, size in sizes.items()}
    tops = {}

    for layer in real_layers:
        cursor = 0
        for node_id in layer:
            tops[node_id] = cursor
            cursor += heights[node_id] + config.node_gap

    for _ in range(max(1, config.coord_iterations)):
        centers = {node_id: tops[node_id] + (heights[node_id] / 2) for node_id in tops}

        for layer_index in range(1, len(real_layers)):
            layer = real_layers[layer_index]
            desired = {
                node_id: median_center(
                    node_id,
                    incoming,
                    centers,
                    centers.get(node_id, 0.0),
                )
                for node_id in layer
            }
            tops.update(place_layer(layer, desired, heights, config.node_gap))

        centers = {node_id: tops[node_id] + (heights[node_id] / 2) for node_id in tops}

        for layer_index in range(len(real_layers) - 2, -1, -1):
            layer = real_layers[layer_index]
            desired = {
                node_id: median_center(
                    node_id,
                    outgoing,
                    centers,
                    centers.get(node_id, 0.0),
                )
                for node_id in layer
            }
            tops.update(place_layer(layer, desired, heights, config.node_gap))

    if tops:
        min_top = min(tops.values())
        shift = config.top_margin - min_top
        for node_id in tops:
            tops[node_id] += shift

    return {node_id: int(round(top)) for node_id, top in tops.items()}


def assign_horizontal_positions(real_layers, sizes, config):
    x_by_layer = {}
    x_cursor = config.left_margin

    for layer_index, layer in enumerate(real_layers):
        layer_width = max((sizes[node_id][0] for node_id in layer), default=0)
        x_by_layer[layer_index] = x_cursor
        x_cursor += layer_width + config.layer_gap

    return x_by_layer


def materialize_dag_layout(canvas, ordered_layers, dummy_ids, incoming, outgoing, sizes, dag_config, shared_config):
    updated = copy.deepcopy(canvas)
    task_nodes = BASE.SHARED.get_tasks(updated)
    task_by_id = {node.get("id"): node for node in task_nodes}
    real_layers = extract_real_layers(ordered_layers, dummy_ids)
    x_by_layer = assign_horizontal_positions(real_layers, sizes, dag_config)
    y_by_node = assign_vertical_positions(real_layers, incoming, outgoing, sizes, dag_config)

    for layer_index, layer in enumerate(real_layers):
        x = x_by_layer[layer_index]
        for node_id in layer:
            node = task_by_id[node_id]
            width, height = sizes[node_id]
            node["x"] = x
            node["y"] = y_by_node[node_id]
            node["width"] = width
            node["height"] = height

    if dag_config.drop_groups:
        kept_nodes = [node for node in updated.get("nodes", []) if node.get("type") != "group"]
        valid_ids = {node.get("id") for node in kept_nodes}
        updated["nodes"] = kept_nodes
        updated["edges"] = [
            edge
            for edge in updated.get("edges", [])
            if edge.get("fromNode") in valid_ids and edge.get("toNode") in valid_ids
        ]

    task_ids = {node.get("id") for node in BASE.SHARED.get_tasks(updated)}
    BASE.optimize_edge_attachment_sides(updated, task_ids, shared_config)
    final_score = BASE.total_layout_score(updated, task_ids, shared_config)

    return updated, real_layers, final_score


def organize_dag(canvas, dag_config, shared_config):
    task_nodes = BASE.SHARED.get_tasks(canvas)
    if not task_nodes:
        raise RuntimeError("Canvas must contain workflow task cards.")

    incoming, outgoing = BASE.build_task_graph(canvas, task_nodes)
    levels = BASE.longest_path_levels(task_nodes, incoming, outgoing)
    sizes = {node.get("id"): BASE.estimate_card_size(node, shared_config) for node in task_nodes}

    proper_layers, proper_parents, proper_children, dummy_ids = build_proper_layers(
        task_nodes,
        levels,
        outgoing,
    )
    median_layers = reduce_crossings(
        proper_layers,
        proper_parents,
        proper_children,
        dag_config.sweep_passes,
    )
    candidate_orders = [median_layers]
    transposed_layers = transpose_layers(median_layers, proper_children)
    if transposed_layers != median_layers:
        candidate_orders.append(transposed_layers)

    best_choice = None
    for ordered_layers in candidate_orders:
        updated, real_layers, final_score = materialize_dag_layout(
            canvas,
            ordered_layers,
            dummy_ids,
            incoming,
            outgoing,
            sizes,
            dag_config,
            shared_config,
        )
        choice = (BASE.layout_score_order(final_score), final_score, updated, real_layers)
        if best_choice is None or choice < best_choice:
            best_choice = choice

    _, final_score, updated, real_layers = best_choice

    return updated, {
        "tasks": len(task_nodes),
        "layers": len(real_layers),
        "occlusion_score": final_score[0],
        "coherence_penalty": final_score[1],
        "port_flow_penalty": final_score[2],
        "edge_crossings": final_score[3],
        "direction_penalty": final_score[4],
        "dropped_groups": dag_config.drop_groups,
    }


def default_output_path(canvas_file):
    path = Path(canvas_file)
    return path.with_name(f"{path.stem}.dag{path.suffix}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pure DAG canvas organizer without group constraints.",
    )
    parser.add_argument("canvas_file", help="Path to the .canvas file to organize")
    parser.add_argument("--output", help="Output path. Defaults to <input>.dag.canvas unless --in-place is used.")
    parser.add_argument("--in-place", action="store_true", help="Rewrite the input canvas in place.")
    parser.add_argument("--keep-groups", action="store_true", help="Keep group nodes in the output canvas.")
    parser.add_argument("--layer-gap", type=int, default=260, help="Horizontal gap between DAG layers.")
    parser.add_argument("--node-gap", type=int, default=60, help="Vertical gap between cards within a layer.")
    parser.add_argument("--sweeps", type=int, default=8, help="Median sweep passes for crossing reduction.")
    parser.add_argument("--coord-iterations", type=int, default=6, help="Coordinate refinement iterations.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.in_place and args.output:
        parser.error("--output and --in-place cannot be used together")

    dag_config = DagLayoutConfig(
        layer_gap=args.layer_gap,
        node_gap=args.node_gap,
        sweep_passes=args.sweeps,
        coord_iterations=args.coord_iterations,
        drop_groups=not args.keep_groups,
    )
    shared_config = BASE.LayoutConfig()

    canvas = BASE.SHARED.load_canvas(args.canvas_file)
    updated, summary = organize_dag(canvas, dag_config, shared_config)

    output_path = Path(args.canvas_file) if args.in_place else Path(args.output or default_output_path(args.canvas_file))
    BASE.SHARED.save_canvas(str(output_path), updated)

    print(f"Organized {summary['tasks']} tasks across {summary['layers']} DAG layers.")
    print(f"Dropped group nodes: {'yes' if summary['dropped_groups'] else 'no'}")
    print(f"Card occlusion score: {summary['occlusion_score']}")
    print(f"Port coherence penalty: {summary['coherence_penalty']}")
    print(f"Port flow penalty: {summary['port_flow_penalty']}")
    print(f"Edge crossing score: {summary['edge_crossings']}")
    print(f"Port direction penalty: {summary['direction_penalty']}")
    print(f"Wrote organized canvas to: {output_path}")


if __name__ == "__main__":
    main()
