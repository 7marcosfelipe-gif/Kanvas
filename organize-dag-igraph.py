#!/usr/bin/env python3
"""No-groups DAG organizer using igraph's Sugiyama layout."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import igraph as ig
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This tool requires python-igraph. Install it with: python -m pip install --user igraph"
    ) from exc


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
class IgraphDagConfig:
    layer_gap: int = 220
    row_gap: int = 70
    margin_x: int = 80
    margin_y: int = 80
    keep_groups: bool = False
    maxiter: int = 200


def build_igraph_graph(task_nodes, outgoing):
    ordered_nodes = sorted(task_nodes, key=BASE.stable_task_key)
    node_ids = [node.get("id") for node in ordered_nodes]
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    graph = ig.Graph(directed=True)
    graph.add_vertices(len(node_ids))
    graph.add_edges(
        [
            (node_index[from_id], node_index[to_id])
            for from_id in node_ids
            for to_id in sorted(outgoing[from_id], key=lambda item: node_index[item])
        ]
    )
    return graph, node_ids, node_index


def layout_with_igraph(task_nodes, levels, outgoing, config):
    graph, node_ids, _ = build_igraph_graph(task_nodes, outgoing)
    ordered_nodes = sorted(task_nodes, key=BASE.stable_task_key)
    layer_vector = [levels[node.get("id")] for node in ordered_nodes]
    layout = graph.layout_sugiyama(
        layers=layer_vector,
        hgap=1,
        vgap=1,
        maxiter=config.maxiter,
    )
    coords = {node_id: tuple(layout[index]) for index, node_id in enumerate(node_ids)}
    return coords


def assign_coordinates(task_nodes, sizes, coords, levels, config):
    layers = {}
    for node in task_nodes:
        node_id = node.get("id")
        layers.setdefault(levels[node_id], []).append(node_id)

    ordered_layers = [layers[layer] for layer in sorted(layers)]
    layer_heights = {
        layer_index: max((sizes[node_id][1] for node_id in layer_nodes), default=0)
        for layer_index, layer_nodes in enumerate(ordered_layers)
    }
    y_by_layer = {}
    y_cursor = config.margin_y
    for layer_index, layer_nodes in enumerate(ordered_layers):
        y_by_layer[layer_index] = y_cursor
        y_cursor += layer_heights[layer_index] + config.layer_gap

    for layer_index, layer_nodes in enumerate(ordered_layers):
        ranked = sorted(
            layer_nodes,
            key=lambda node_id: (coords[node_id][0], BASE.stable_task_key(next(node for node in task_nodes if node.get("id") == node_id))),
        )
        x_cursor = config.margin_x
        for node_id in ranked:
            width, height = sizes[node_id]
            node = next(node for node in task_nodes if node.get("id") == node_id)
            node["x"] = int(x_cursor)
            node["y"] = int(y_by_layer[layer_index] + max(0, (layer_heights[layer_index] - height) / 2))
            node["width"] = width
            node["height"] = height
            x_cursor += width + config.row_gap


def organize_dag(canvas, igraph_config, shared_config):
    updated = copy.deepcopy(canvas)
    task_nodes = BASE.SHARED.get_tasks(updated)
    if not task_nodes:
        raise RuntimeError("Canvas must contain workflow task cards.")

    incoming, outgoing = BASE.build_task_graph(updated, task_nodes)
    levels = BASE.longest_path_levels(task_nodes, incoming, outgoing)
    sizes = {node.get("id"): BASE.estimate_card_size(node, shared_config) for node in task_nodes}
    coords = layout_with_igraph(task_nodes, levels, outgoing, igraph_config)
    assign_coordinates(task_nodes, sizes, coords, levels, igraph_config)

    if not igraph_config.keep_groups:
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
    score = BASE.total_layout_score(updated, task_ids, shared_config)

    return updated, {
        "tasks": len(task_nodes),
        "layers": max(levels.values(), default=-1) + 1,
        "occlusion_score": score[0],
        "coherence_penalty": score[1],
        "port_flow_penalty": score[2],
        "edge_crossings": score[3],
        "direction_penalty": score[4],
        "keep_groups": igraph_config.keep_groups,
    }


def default_output_path(canvas_file):
    path = Path(canvas_file)
    return path.with_name(f"{path.stem}.igraph-dag{path.suffix}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="No-groups DAG organizer using igraph Sugiyama layout.",
    )
    parser.add_argument("canvas_file", help="Path to the .canvas file to organize")
    parser.add_argument("--output", help="Output path. Defaults to <input>.igraph-dag.canvas unless --in-place is used.")
    parser.add_argument("--in-place", action="store_true", help="Rewrite the input canvas in place.")
    parser.add_argument("--keep-groups", action="store_true", help="Keep group nodes in the output.")
    parser.add_argument("--layer-gap", type=int, default=220, help="Vertical gap between DAG layers.")
    parser.add_argument("--row-gap", type=int, default=70, help="Horizontal gap between cards within a layer.")
    parser.add_argument("--maxiter", type=int, default=200, help="Sugiyama crossing-reduction iterations in igraph.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.in_place and args.output:
        parser.error("--output and --in-place cannot be used together")

    config = IgraphDagConfig(
        layer_gap=args.layer_gap,
        row_gap=args.row_gap,
        keep_groups=args.keep_groups,
        maxiter=args.maxiter,
    )
    shared_config = BASE.LayoutConfig()

    canvas = BASE.SHARED.load_canvas(args.canvas_file)
    updated, summary = organize_dag(canvas, config, shared_config)

    output_path = Path(args.canvas_file) if args.in_place else Path(args.output or default_output_path(args.canvas_file))
    BASE.SHARED.save_canvas(str(output_path), updated)

    print(f"Organized {summary['tasks']} tasks across {summary['layers']} DAG layers.")
    print(f"Kept group nodes: {'yes' if summary['keep_groups'] else 'no'}")
    print(f"Card occlusion score: {summary['occlusion_score']}")
    print(f"Port coherence penalty: {summary['coherence_penalty']}")
    print(f"Port flow penalty: {summary['port_flow_penalty']}")
    print(f"Edge crossing score: {summary['edge_crossings']}")
    print(f"Port direction penalty: {summary['direction_penalty']}")
    print(f"Wrote organized canvas to: {output_path}")


if __name__ == "__main__":
    main()
