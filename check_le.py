import json
from collections import defaultdict

c = json.load(open('layout-fixtures/stress-test.dag-groups.canvas'))
tasks = {n['id']: n for n in c['nodes'] if n.get('type') == 'text'}
groups = [n for n in c['nodes'] if n.get('type') == 'group']
edges = c['edges']

print(f"Output: {len(tasks)} tasks, {len(groups)} groups, {len(edges)} edges")
print()
print("Groups:")
for g in sorted(groups, key=lambda g: (g['x'], g['y'])):
    print(f"  {g['label']:20s} x={g['x']:5d} y={g['y']:6d} w={g['width']:5d} h={g['height']:5d}")
print()
print("Cards per row (y):")
rows = defaultdict(list)
for n in tasks.values():
    rows[n['y']].append(n['text'].split('\n')[0][:25])
for y in sorted(rows):
    labels = ', '.join(sorted(rows[y]))
    print(f"  y={y:6d}: {labels}")
