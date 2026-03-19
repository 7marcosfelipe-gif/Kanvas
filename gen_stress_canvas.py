#!/usr/bin/env python3
"""Generate a stress-test canvas with cross-group deps, long chains, fan-in/out, and disjoint subgraphs."""
import json, random, string

def uid(prefix=""):
    chars = string.ascii_lowercase + string.digits
    return prefix + ''.join(random.choices(chars, k=8))

random.seed(42)

# ── task definitions ─────────────────────────────────────────────────────────
# (id_key, label, group, description)
TASKS = [
    # Infrastructure  (connected main graph, depth 0-1)
    ("INF-01", "Provision base VMs",           "infra",    "Set up virtual machines and networking."),
    ("INF-02", "Configure DNS and load balancer", "infra", "Register domains, set up LB rules."),
    ("INF-03", "Bootstrap secrets manager",    "infra",    "Vault init, policies, auth backends."),
    # Backend  (depth 2-4)
    ("BE-01",  "Scaffold API service",          "backend",  "Generate OpenAPI stubs, wire DI."),
    ("BE-02",  "Implement auth endpoints",      "backend",  "JWT issue, refresh, revoke flows."),
    ("BE-03",  "Implement data CRUD",           "backend",  "REST handlers with DB transactions."),
    ("BE-04",  "Add rate limiting & caching",   "backend",  "Redis-backed throttle + cache layer."),
    ("BE-05",  "Write integration tests",       "backend",  "Cover all happy and error paths."),
    # Frontend  (depth 2-5, overlaps with Backend in depth)
    ("FE-01",  "Bootstrap SPA shell",           "frontend", "Vite + React + router setup."),
    ("FE-02",  "Build auth UI",                 "frontend", "Login, logout, token refresh UI."),
    ("FE-03",  "Build data dashboard",          "frontend", "Tables, charts, filter controls."),
    ("FE-04",  "Cross-browser testing",         "frontend", "Playwright suite on Chrome/FF/Safari."),
    ("FE-05",  "Accessibility audit",           "frontend", "WCAG 2.1 AA automated + manual."),
    # Testing  (depth 3-7, overlaps with BE and FE)
    ("TE-01",  "Define test strategy",         "testing",  "Scope, tooling, coverage targets."),
    ("TE-02",  "API contract tests",            "testing",  "Pact broker for BE↔FE contracts."),
    ("TE-03",  "E2E smoke suite",               "testing",  "Critical paths on staging env."),
    ("TE-04",  "Performance baseline",          "testing",  "k6 load test, capture p50/p99."),
    ("TE-05",  "Security scan",                 "testing",  "OWASP ZAP + Trivy on images."),
    # Deployment  (depth 5-9)
    ("DP-01",  "Containerise services",         "deploy",   "Dockerfiles + compose for all svcs."),
    ("DP-02",  "Write Helm charts",             "deploy",   "K8s manifests, values.yaml, secrets."),
    ("DP-03",  "CI pipeline",                   "deploy",   "GHA workflow: lint, test, build, push."),
    ("DP-04",  "Canary rollout procedure",      "deploy",   "Argo Rollouts config, analysis runs."),
    # Documentation  ── DISJOINT subgraph ──
    ("DO-01",  "Architecture decision records", "docs",     "Document key tech choices (ADRs)."),
    ("DO-02",  "API reference docs",            "docs",     "OpenAPI → Redoc, hosted internally."),
    ("DO-03",  "Runbook: on-call guide",        "docs",     "Incident playbooks, escalation paths."),
    ("DO-04",  "Changelog & release notes",     "docs",     "Keep-a-changelog format, semver tags."),
    # Analytics  ── second DISJOINT subgraph (longer chain) ──
    ("AN-01",  "Define KPI taxonomy",           "analytics","Metric names, dimensions, owners."),
    ("AN-02",  "Instrument frontend events",    "analytics","Segment.io calls on key user actions."),
    ("AN-03",  "Build data pipeline",           "analytics","Kafka → Flink → Clickhouse ETL."),
    ("AN-04",  "Create dashboards",             "analytics","Grafana boards: DAU, retention, perf."),
    ("AN-05",  "Alerting & anomaly rules",      "analytics","PagerDuty integration, threshold alerts."),
]

# ── edge definitions (src_key → dst_key) ──────────────────────────────────────
EDGES_KEYS = [
    # INF internal
    ("INF-01", "INF-03"),
    ("INF-02", "INF-03"),
    # INF → BE  (cross-group)
    ("INF-01", "BE-01"),
    ("INF-03", "BE-01"),
    # INF → FE  (cross-group)
    ("INF-02", "FE-01"),
    ("INF-03", "FE-01"),
    # BE internal
    ("BE-01", "BE-02"),
    ("BE-01", "BE-03"),
    ("BE-02", "BE-04"),
    ("BE-03", "BE-04"),
    ("BE-04", "BE-05"),
    # FE internal
    ("FE-01", "FE-02"),
    ("FE-01", "FE-03"),
    ("FE-02", "FE-04"),
    ("FE-03", "FE-04"),
    ("FE-04", "FE-05"),
    # BE → FE  cross-group (creates depth conflict: FE-03 must be after BE-02)
    ("BE-02", "FE-03"),
    # BE → TE  cross-group fan-in
    ("BE-01", "TE-01"),
    ("BE-04", "TE-02"),
    ("BE-05", "TE-03"),
    # FE → TE  cross-group
    ("FE-01", "TE-01"),
    ("FE-04", "TE-03"),
    ("FE-05", "TE-04"),
    # TE internal
    ("TE-01", "TE-02"),
    ("TE-01", "TE-03"),
    ("TE-02", "TE-04"),
    ("TE-03", "TE-04"),
    ("TE-04", "TE-05"),
    # BE → DP  cross-group
    ("BE-04", "DP-01"),
    ("BE-05", "DP-03"),
    # FE → DP  cross-group
    ("FE-04", "DP-01"),
    # DP internal
    ("DP-01", "DP-02"),
    ("DP-02", "DP-03"),
    ("DP-03", "DP-04"),
    # TE → DP  cross-group (long cross-group chain)
    ("TE-05", "DP-04"),
    # Documentation  (disjoint)
    ("DO-01", "DO-02"),
    ("DO-01", "DO-03"),
    ("DO-02", "DO-04"),
    ("DO-03", "DO-04"),
    # Analytics  (disjoint, linear chain)
    ("AN-01", "AN-02"),
    ("AN-01", "AN-03"),
    ("AN-02", "AN-04"),
    ("AN-03", "AN-04"),
    ("AN-04", "AN-05"),
]

# ── group initial geometry (cards will be placed inside) ──────────────────────
GROUP_META = {
    "infra":     {"label": "Infrastructure", "col": 0, "row": 0},
    "backend":   {"label": "Backend",         "col": 1, "row": 0},
    "frontend":  {"label": "Frontend",        "col": 2, "row": 0},
    "testing":   {"label": "Testing",         "col": 0, "row": 1},
    "deploy":    {"label": "Deployment",      "col": 1, "row": 1},
    "docs":      {"label": "Documentation",   "col": 2, "row": 1},
    "analytics": {"label": "Analytics",       "col": 0, "row": 2},
}

COL_W, ROW_H = 1400, 1200
PAD_X, PAD_Y = 80, 120
CARD_W, CARD_H = 300, 180
CARD_GAP = 40

# assign node IDs
id_map = {key: uid() for key, *_ in TASKS}

nodes = []
edges = []

# group nodes + card nodes
for gkey, gmeta in GROUP_META.items():
    gx = gmeta["col"] * COL_W
    gy = gmeta["row"] * ROW_H
    members = [(key, lbl, desc) for key, lbl, grp, desc in TASKS if grp == gkey]
    cols_per_group = 2
    gw = PAD_X * 2 + cols_per_group * CARD_W + (cols_per_group - 1) * CARD_GAP
    gh = PAD_Y + len(members) * (CARD_H + CARD_GAP) + 40

    gid = "group-" + gkey
    nodes.append({
        "id": gid, "type": "group", "label": gmeta["label"],
        "x": gx, "y": gy, "width": gw, "height": gh
    })

    for i, (key, lbl, desc) in enumerate(members):
        cx = gx + PAD_X
        cy = gy + PAD_Y + i * (CARD_H + CARD_GAP)
        text = f"## {key} {lbl}\n\n{desc}"
        nodes.append({
            "id": id_map[key], "type": "text", "text": text,
            "x": cx, "y": cy, "width": CARD_W, "height": CARD_H
        })

# edges
for src, dst in EDGES_KEYS:
    edges.append({
        "id": uid("e"),
        "fromNode": id_map[src], "toNode": id_map[dst],
        "fromSide": "bottom", "toSide": "top"
    })

nodes.append({
    "id": "legend", "type": "text",
    "text": "## Legend\n🔴 **Red** = To Do (ready)\n⬛ **Gray** = Blocked\n🟠 **Orange** = Doing\n🔵 **Cyan** = Ready to review\n🟢 **Green** = Done\n🟣 **Purple** = Proposed by agent",
    "x": -540, "y": -310, "width": 380, "height": 230, "color": "6"
})

canvas = {"nodes": nodes, "edges": edges}

out = "layout-fixtures/stress-test.canvas"
with open(out, "w", encoding="utf-8") as f:
    json.dump(canvas, f, indent="\t", ensure_ascii=False)

print(f"Created {out}")
print(f"  {len(nodes) - len(GROUP_META)} task cards, {len(GROUP_META)} groups, {len(edges)} edges")
print(f"  Groups: {', '.join(g['label'] for g in GROUP_META.values())}")
print(f"  Disjoint subgraphs: Documentation, Analytics")
