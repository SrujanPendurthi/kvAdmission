"""Recall vs distance-back, per policy per budget. The horizon is where it cliffs."""
import json, sys
from collections import defaultdict

rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results.json"))
dists = sorted({r["distance_back"] for r in rows})
grid = defaultdict(dict)
for r in rows:
    grid[(r["policy"], r["budget"])][r["distance_back"]] = r["hit"]

print("recall by distance-back (near -> far)\n")
print(f"{'policy':8s} {'B':>5s} | " + " ".join(f"{d:>5d}" for d in dists) + " | rate")
print("-" * (16 + 6 * len(dists) + 8))
for (pol, b), hits in sorted(grid.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    cells = " ".join("    +" if hits.get(d) else "    ." for d in dists)
    rate = sum(hits.values()) / len(hits)
    print(f"{pol:8s} {b:>5d} | {cells} | {rate:.2f}")

print("\nhorizon = largest distance still recalled")
for (pol, b), hits in sorted(grid.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    got = [d for d, h in hits.items() if h]
    print(f"  {pol:8s} B={b:<5d} horizon={max(got) if got else 0:<6d} "
          f"ratio_to_budget={(max(got)/b if got else 0):.2f}")
