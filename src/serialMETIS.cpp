#include <iostream>
#include <vector>
#include <limits>
#include <queue>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <metis.h> // METIS library for graph partitioning

using namespace std;

const float INF = numeric_limits<float>::max();

struct Graph {
    int num_vertices;
    vector<vector<pair<int, float>>> adj_list; // adjacency list: [u] -> [(v, weight)]
    vector<int> partitions; // METIS partition assignments for each vertex
    
    // SSSP tree data structures
    vector<float> distance;    // Dist array
    vector<int> parent;        // Parent array
    vector<bool> affected;     // Affected array
    vector<bool> affected_del; // Affected_Del array
    
    Graph(int n) : num_vertices(n), 
                  adj_list(n),
                  partitions(n, -1),
                  distance(n, INF),
                  parent(n, -1),
                  affected(n, false),
                  affected_del(n, false) {}
};

Graph read_mtx_graph(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    string line;
    
    // Check if the file follows MTX format or simple edge list format
    getline(file, line);
    file.seekg(0); // Reset file position to start
    
    if (line[0] == '%') {
        // MTX format - Skip comments
        while (getline(file, line)) {
            if (line[0] != '%') break;
        }
        
        // Read matrix dimensions
        int num_vertices, num_edges;
        istringstream iss(line);
        iss >> num_vertices >> num_vertices >> num_edges;
        
        Graph graph(num_vertices);
        
        // Read edges
        for (int i = 0; i < num_edges; ++i) {
            getline(file, line);
            if (line.empty()) continue;
            
            int u, v;
            istringstream edge_stream(line);
            edge_stream >> u >> v;
            
            // MTX uses 1-based indexing, convert to 0-based
            u--; v--;
            
            // Add edge with default weight 1.0 (pattern format)
            graph.adj_list[u].emplace_back(v, 1.0f);
            graph.adj_list[v].emplace_back(u, 1.0f); // symmetric
        }
        
        return graph;
    } else {
        // Simple edge list format: first line contains number of vertices and edges
        int num_vertices, num_edges;
        istringstream iss(line);
        iss >> num_vertices >> num_edges;
        
        Graph graph(num_vertices);
        
        // Read edges with weights
        string edge_line;
        while (getline(file, edge_line)) {
            if (edge_line.empty()) continue;
            
            istringstream edge_stream(edge_line);
            int u, v;
            float weight = 1.0f; // Default weight if not specified
            
            edge_stream >> u >> v;
            if (edge_stream >> weight) {
                // Weight was specified
            }
            
            // Add edge (assuming 0-based indexing in input)
            if (u >= 0 && u < num_vertices && v >= 0 && v < num_vertices) {
                graph.adj_list[u].emplace_back(v, weight);
                graph.adj_list[v].emplace_back(u, weight); // assuming undirected graph
            }
        }
        
        return graph;
    }
}

// Function to partition the graph using METIS
void partition_graph(Graph& graph, int num_partitions) {
    if (num_partitions <= 1) {
        // No partitioning needed
        fill(graph.partitions.begin(), graph.partitions.end(), 0);
        return;
    }

    // METIS variables
    idx_t nvtxs = graph.num_vertices;
    idx_t ncon = 1; // number of balancing constraints
    idx_t nparts = num_partitions; // number of partitions
    idx_t objval; // objective value
    vector<idx_t> part(graph.num_vertices); // partition array
    
    // Convert adjacency list to METIS format
    vector<idx_t> xadj;
    vector<idx_t> adjncy;
    vector<idx_t> adjwgt; // edge weights (optional)
    
    xadj.push_back(0);
    for (int u = 0; u < graph.num_vertices; u++) {
        for (const auto& edge : graph.adj_list[u]) {
            adjncy.push_back(edge.first);
            // If using weights (optional)
            adjwgt.push_back(static_cast<idx_t>(edge.second * 1000)); // scale float to integer
        }
        xadj.push_back(adjncy.size());
    }
    
    // METIS options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT; // minimize edge-cut
    options[METIS_OPTION_CONTIG] = 1; // force contiguous partitions
    
    // Call METIS
    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                NULL, NULL, adjwgt.data(), &nparts, NULL,
                                NULL, options, &objval, part.data());
    
    if (ret != METIS_OK) {
        cerr << "METIS partitioning failed!" << endl;
        exit(1);
    }
    
    // Store partitions in the graph structure
    for (int i = 0; i < graph.num_vertices; i++) {
        graph.partitions[i] = part[i];
    }
    
    cout << "Graph partitioned into " << num_partitions << " parts with edge-cut: " << objval << endl;
}

void process_changed_edges(Graph& graph, 
                         const vector<pair<pair<int, int>, float>>& insertions,
                         const vector<pair<int, int>>& deletions) {
    
    fill(graph.affected.begin(), graph.affected.end(), false);
    fill(graph.affected_del.begin(), graph.affected_del.end(), false);
    
    // Process deletions
    for (const auto& edge : deletions) {
        int u = edge.first;
        int v = edge.second;
        
        if (graph.parent[v] == u || graph.parent[u] == v) {
            int y = (graph.distance[u] > graph.distance[v]) ? u : v;
            graph.distance[y] = INF;
            graph.parent[y] = -1;
            graph.affected_del[y] = true;
            graph.affected[y] = true;
        }
    }
    
    // Process insertions
    for (const auto& edge : insertions) {
        int u = edge.first.first;
        int v = edge.first.second;
        float weight = edge.second;
        
        int x = (graph.distance[u] < graph.distance[v]) ? u : v;
        int y = (x == u) ? v : u;
        
        if (graph.distance[y] > graph.distance[x] + weight) {
            graph.distance[y] = graph.distance[x] + weight;
            graph.parent[y] = x;
            graph.affected[y] = true;
        }
    }
}

void update_affected_vertices(Graph& graph) {
    // Part 1: Process deletion-affected vertices
    bool changed = true;
    while (changed) {
        changed = false;
        
        for (int v = 0; v < graph.num_vertices; v++) {
            if (graph.affected_del[v]) {
                graph.affected_del[v] = false;
                
                for (int c = 0; c < graph.num_vertices; c++) {
                    if (graph.parent[c] == v) {
                        graph.distance[c] = INF;
                        graph.parent[c] = -1;
                        graph.affected_del[c] = true;
                        graph.affected[c] = true;
                        changed = true;
                    }
                }
            }
        }
    }
    
    // Part 2: Update distances of affected vertices
    changed = true;
    while (changed) {
        changed = false;
        
        for (int v = 0; v < graph.num_vertices; v++) {
            if (graph.affected[v]) {
                graph.affected[v] = false;
                
                // Only process neighbors in the same partition (optional optimization)
                for (const auto& neighbor : graph.adj_list[v]) {
                    int n = neighbor.first;
                    float weight = neighbor.second;
                    
                    if (graph.distance[n] > graph.distance[v] + weight) {
                        graph.distance[n] = graph.distance[v] + weight;
                        graph.parent[n] = v;
                        graph.affected[n] = true;
                        changed = true;
                    }
                    
                    if (graph.distance[v] > graph.distance[n] + weight) {
                        graph.distance[v] = graph.distance[n] + weight;
                        graph.parent[v] = n;
                        graph.affected[v] = true;
                        changed = true;
                    }
                }
            }
        }
    }
}

void initialize_sssp(Graph& graph, int source) {
    fill(graph.distance.begin(), graph.distance.end(), INF);
    fill(graph.parent.begin(), graph.parent.end(), -1);
    
    graph.distance[source] = 0;
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> pq;
    pq.push({0, source});
    
    while (!pq.empty()) {
        auto [dist_u, u] = pq.top();
        pq.pop();
        
        if (dist_u > graph.distance[u]) continue;
        
        for (const auto& [v, weight] : graph.adj_list[u]) {
            if (graph.distance[v] > graph.distance[u] + weight) {
                graph.distance[v] = graph.distance[u] + weight;
                graph.parent[v] = u;
                pq.push({graph.distance[v], v});
            }
        }
    }
}

void update_sssp(Graph& graph, 
                const vector<pair<pair<int, int>, float>>& insertions,
                const vector<pair<int, int>>& deletions) {
    cout << "\nProcessing updates..." << endl;
    
    // Display update details
    if (!insertions.empty()) {
        cout << "Insertions:" << endl;
        for (const auto& edge : insertions) {
            cout << "  Adding edge: " << edge.first.first << " -- " 
                 << edge.first.second << " (weight: " << edge.second << ")" << endl;
        }
    }
    
    if (!deletions.empty()) {
        cout << "Deletions:" << endl;
        for (const auto& edge : deletions) {
            cout << "  Removing edge: " << edge.first << " -- " << edge.second << endl;
        }
    }
    
    // Process the updates
    process_changed_edges(graph, insertions, deletions);
    update_affected_vertices(graph);
    
    cout << "Update completed." << endl;
}

void print_sssp_table(const Graph& graph, int source) {
    cout << "\nCurrent shortest paths from source " << source << ":" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Vertex | Distance | Parent | Partition" << endl;
    cout << "----------------------------------------" << endl;
    
    if (graph.num_vertices <= 15) {
        for (int v = 0; v < graph.num_vertices; v++) {
            cout << setw(6) << left << v << " | ";
            if (graph.distance[v] == INF)
                cout << setw(8) << left << "INF";
            else
                cout << setw(8) << left << graph.distance[v];
            cout << " | ";
            
            if (graph.parent[v] == -1)
                cout << setw(6) << left << "-";
            else
                cout << setw(6) << left << graph.parent[v];
            
            cout << " | " << graph.partitions[v] << endl;
        }
    } else {
        cout << "The graph has " << graph.num_vertices << " vertices." << endl;
        cout << "How would you like to view the results?" << endl;
        cout << "1. Show first N vertices" << endl;
        cout << "2. Show specific vertices (comma-separated)" << endl;
        cout << "3. Show all vertices" << endl;
        cout << "4. Export results to file" << endl;
        cout << "Enter your choice (1-4): ";
        
        int choice;
        cin >> choice;
        
        switch (choice) {
            case 1: {
                int n;
                cout << "Enter number of vertices to show: ";
                cin >> n;
                n = min(n, graph.num_vertices);
                
                for (int v = 0; v < n; v++) {
                    cout << setw(6) << left << v << " | ";
                    if (graph.distance[v] == INF)
                        cout << setw(8) << left << "INF";
                    else
                        cout << setw(8) << left << graph.distance[v];
                    cout << " | ";
                    
                    if (graph.parent[v] == -1)
                        cout << setw(6) << left << "-";
                    else
                        cout << setw(6) << left << graph.parent[v];
                    
                    cout << " | " << graph.partitions[v] << endl;
                }
                break;
            }
            case 2: {
                cout << "Enter vertices to show (comma-separated, e.g., 0,1,5,10): ";
                cin.ignore();
                string input;
                getline(cin, input);
                
                stringstream ss(input);
                string vertex_str;
                while (getline(ss, vertex_str, ',')) {
                    try {
                        int v = stoi(vertex_str);
                        if (v >= 0 && v < graph.num_vertices) {
                            cout << setw(6) << left << v << " | ";
                            if (graph.distance[v] == INF)
                                cout << setw(8) << left << "INF";
                            else
                                cout << setw(8) << left << graph.distance[v];
                            cout << " | ";
                            
                            if (graph.parent[v] == -1)
                                cout << setw(6) << left << "-";
                            else
                                cout << setw(6) << left << graph.parent[v];
                            
                            cout << " | " << graph.partitions[v] << endl;
                        } else {
                            cout << "Vertex " << v << " is out of range." << endl;
                        }
                    } catch (const exception& e) {
                        cout << "Invalid input: " << vertex_str << endl;
                    }
                }
                break;
            }
            case 3: {
                cout << "Displaying all " << graph.num_vertices << " vertices:" << endl;
                for (int v = 0; v < graph.num_vertices; v++) {
                    cout << setw(6) << left << v << " | ";
                    if (graph.distance[v] == INF)
                        cout << setw(8) << left << "INF";
                    else
                        cout << setw(8) << left << graph.distance[v];
                    cout << " | ";
                    
                    if (graph.parent[v] == -1)
                        cout << setw(6) << left << "-";
                    else
                        cout << setw(6) << left << graph.parent[v];
                    
                    cout << " | " << graph.partitions[v] << endl;
                }
                break;
            }
            case 4: {
                string filename;
                cout << "Enter output filename: ";
                cin >> filename;
                
                ofstream outfile(filename);
                if (!outfile.is_open()) {
                    cerr << "Error: Could not open file for writing." << endl;
                    break;
                }
                
                outfile << "Vertex,Distance,Parent,Partition" << endl;
                for (int v = 0; v < graph.num_vertices; v++) {
                    outfile << v << ",";
                    if (graph.distance[v] == INF)
                        outfile << "INF";
                    else
                        outfile << graph.distance[v];
                    outfile << ",";
                    
                    if (graph.parent[v] == -1)
                        outfile << "-";
                    else
                        outfile << graph.parent[v];
                    
                    outfile << "," << graph.partitions[v] << endl;
                }
                
                outfile.close();
                cout << "Results saved to " << filename << endl;
                break;
            }
            default:
                cout << "Invalid choice. Showing first 10 vertices by default." << endl;
                for (int v = 0; v < min(10, graph.num_vertices); v++) {
                    cout << setw(6) << left << v << " | ";
                    if (graph.distance[v] == INF)
                        cout << setw(8) << left << "INF";
                    else
                        cout << setw(8) << left << graph.distance[v];
                    cout << " | ";
                    
                    if (graph.parent[v] == -1)
                        cout << setw(6) << left << "-";
                    else
                        cout << setw(6) << left << graph.parent[v];
                    
                    cout << " | " << graph.partitions[v] << endl;
                }
        }
    }
    
    cout << "----------------------------------------" << endl;
}

void build_tree_representation(const Graph& graph, int node, vector<string>& lines, string prefix = "") {
    vector<int> children;
    for (int v = 0; v < graph.num_vertices; v++) {
        if (graph.parent[v] == node && v != node) {
            children.push_back(v);
        }
    }
    
    sort(children.begin(), children.end(), [&graph](int a, int b) {
        return graph.distance[a] < graph.distance[b];
    });
    
    if (node >= 0) {
        string line = prefix + "└── " + to_string(node) + " (" + 
                     (graph.distance[node] == INF ? "INF" : to_string((int)graph.distance[node])) + 
                     ") [P" + to_string(graph.partitions[node]) + "]";
        lines.push_back(line);
        prefix += "    ";
    }
    
    for (size_t i = 0; i < children.size(); i++) {
        build_tree_representation(graph, children[i], lines, prefix);
    }
}

void print_shortest_path_tree(const Graph& graph, int source) {
    cout << "\nShortest Path Tree from source " << source << ":" << endl;
    cout << "----------------------------------------" << endl;
    
    vector<string> tree_lines;
    build_tree_representation(graph, source, tree_lines, "");
    
    for (const auto& line : tree_lines) {
        cout << line << endl;
    }
    
    cout << "----------------------------------------" << endl;
}

void print_partition_stats(const Graph& graph) {
    if (graph.partitions.empty() || *max_element(graph.partitions.begin(), graph.partitions.end()) < 0) {
        cout << "Graph has not been partitioned yet." << endl;
        return;
    }
    
    int num_partitions = *max_element(graph.partitions.begin(), graph.partitions.end()) + 1;
    vector<int> partition_sizes(num_partitions, 0);
    
    for (int p : graph.partitions) {
        partition_sizes[p]++;
    }
    
    cout << "\nPartition Statistics:" << endl;
    cout << "Total partitions: " << num_partitions << endl;
    for (int i = 0; i < num_partitions; i++) {
        cout << "Partition " << i << ": " << partition_sizes[i] << " vertices ("
             << fixed << setprecision(1) << (100.0 * partition_sizes[i] / graph.num_vertices)
             << "%)" << endl;
    }
    
    // Calculate edge-cut (approximate)
    int edge_cut = 0;
    for (int u = 0; u < graph.num_vertices; u++) {
        for (const auto& edge : graph.adj_list[u]) {
            int v = edge.first;
            if (graph.partitions[u] != graph.partitions[v]) {
                edge_cut++;
            }
        }
    }
    edge_cut /= 2; // Each edge counted twice
    
    cout << "Edge-cut between partitions: " << edge_cut << " edges" << endl;
}

int main() {
    string filename;
    cout << "Enter graph file path (or press Enter to use default 'graph.txt'): ";
    getline(cin, filename);
    
    if (filename.empty()) {
        filename = "graph.txt";
    }
    
    cout << "Reading graph from " << filename << endl;
    Graph graph = read_mtx_graph(filename);
    cout << "Graph loaded with " << graph.num_vertices << " vertices" << endl;

    // Partition the graph
    int num_partitions;
    cout << "\nEnter number of partitions (1 for no partitioning): ";
    cin >> num_partitions;
    
    if (num_partitions > 1) {
        partition_graph(graph, num_partitions);
        print_partition_stats(graph);
    } else {
        cout << "Skipping partitioning." << endl;
    }

    // Ask user for source vertex
    int source;
    cout << "\nEnter source vertex (0-" << graph.num_vertices-1 << "): ";
    cin >> source;
    
    if (source < 0 || source >= graph.num_vertices) {
        cout << "Invalid source vertex. Using default source 0." << endl;
        source = 0;
    }

    cout << "\nComputing initial SSSP from vertex " << source << endl;
    initialize_sssp(graph, source);
    
    print_sssp_table(graph, source);
    print_shortest_path_tree(graph, source);

    // Optional: Process updates
    char apply_updates;
    cout << "\nApply graph updates? (y/n): ";
    cin >> apply_updates;
    
    if (apply_updates == 'y' || apply_updates == 'Y') {
        vector<pair<pair<int, int>, float>> insertions;
        vector<pair<int, int>> deletions;
        
        char more = 'y';
        
        cout << "\nEnter edge insertions:" << endl;
        while (more == 'y' || more == 'Y') {
            int u, v;
            float weight;
            
            cout << "Enter source vertex: ";
            cin >> u;
            cout << "Enter destination vertex: ";
            cin >> v;
            cout << "Enter weight: ";
            cin >> weight;
            
            if (u >= 0 && u < graph.num_vertices && v >= 0 && v < graph.num_vertices) {
                insertions.push_back({{u, v}, weight});
                cout << "Edge " << u << " -- " << v << " (weight: " << weight << ") added for insertion." << endl;
            } else {
                cout << "Invalid vertices. Edge not added." << endl;
            }
            
            cout << "Add another insertion? (y/n): ";
            cin >> more;
        }
        
        more = 'y';
        cout << "\nEnter edge deletions:" << endl;
        while (more == 'y' || more == 'Y') {
            int u, v;
            
            cout << "Enter source vertex: ";
            cin >> u;
            cout << "Enter destination vertex: ";
            cin >> v;
            
            if (u >= 0 && u < graph.num_vertices && v >= 0 && v < graph.num_vertices) {
                deletions.push_back({u, v});
                cout << "Edge " << u << " -- " << v << " added for deletion." << endl;
            } else {
                cout << "Invalid vertices. Edge not added." << endl;
            }
            
            cout << "Add another deletion? (y/n): ";
            cin >> more;
        }

        update_sssp(graph, insertions, deletions);
        
        cout << "\nAfter updates:" << endl;
        print_sssp_table(graph, source);
        print_shortest_path_tree(graph, source);
    }

    return 0;
}
