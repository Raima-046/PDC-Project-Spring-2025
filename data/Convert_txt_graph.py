import os
import subprocess
from collections import defaultdict

def convert_to_graph(input_file, output_file):
    adj = defaultdict(list)
    edge_set = set()

    # Step 1: Build adjacency list and count unique edges
    with open(input_file, 'r') as f:
        for line in f:
            u, v, w = map(int, line.strip().split())
            if u > v:
                u, v = v, u  # to avoid double-counting undirected edges
            edge_set.add((u, v, w))
            adj[u].append((v, w))
            adj[v].append((u, w))

    num_nodes = max(adj.keys()) + 1
    num_edges = len(edge_set)

    # Step 2: Write to METIS format file
    with open(output_file, 'w') as f:
        f.write(f"{num_nodes} {num_edges} 1\n")  # "1" means edge weights included
        for i in range(num_nodes):
            neighbors = adj.get(i, [])
            line = ' '.join(f"{v + 1} {w}" for v, w in neighbors)  # convert to 1-based
            f.write(line + "\n")

    print(f"[✓] Converted '{input_file}' to METIS format as '{output_file}'.")


def run_gpmetis(graph_file, num_parts=2):
    result = subprocess.run(["gpmetis", graph_file, str(num_parts)], capture_output=True, text=True)

    if result.returncode != 0:
        print("[✗] gpmetis failed to run:")
        print(result.stderr)
    else:
        print(f"[✓] gpmetis partitioned the graph into {num_parts} parts.")


def count_partitions(partition_file):
    counts = {}

    with open(partition_file, 'r') as f:
        for line in f:
            part = int(line.strip())
            counts[part] = counts.get(part, 0) + 1

    print("\nPartition Summary:")
    for part in sorted(counts.keys()):
        print(f"  Part {part}: {counts[part]} nodes")


# === Main Execution ===
input_txt = "graph.txt"
output_graph = "graph.graph"
num_parts = 2
partition_output = f"{output_graph}.part.{num_parts}"

convert_to_graph(input_txt, output_graph)
run_gpmetis(output_graph, num_parts)

if os.path.exists(partition_output):
    count_partitions(partition_output)
else:
    print(f"[✗] Partition file '{partition_output}' not found.")

