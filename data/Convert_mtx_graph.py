def convert_mtx_to_graph(mtx_file, graph_file, directed=False):
    with open(mtx_file, 'r') as fin:
        lines = fin.readlines()

    # Skip comments
    data_lines = [line.strip() for line in lines if not line.startswith('%') and line.strip()]
    
    # First non-comment line is header
    header = data_lines[0]
    num_rows, num_cols, num_edges = map(int, header.split())

    # Initialize adjacency list
    adjacency = {i: [] for i in range(1, num_rows + 1)}

    for line in data_lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            u, v = int(parts[0]), int(parts[1])
            adjacency[u].append(v)
            if not directed:
                adjacency[v].append(u)

    # Compute actual edge count (undirected edges counted once)
    if directed:
        edge_count = sum(len(neighs) for neighs in adjacency.values())
    else:
        edge_count = sum(len(neighs) for neighs in adjacency.values()) // 2

    with open(graph_file, 'w') as fout:
        fout.write(f"{num_rows} {edge_count}\n")
        for node in range(1, num_rows + 1):
            neighbors = " ".join(str(neigh) for neigh in adjacency[node])
            fout.write(neighbors + "\n")

    print(f"Converted {num_rows} vertices and {edge_count} edges to {graph_file}")

# Example usage
convert_mtx_to_graph('dataset.mtx', 'dataset.graph', directed=False)
