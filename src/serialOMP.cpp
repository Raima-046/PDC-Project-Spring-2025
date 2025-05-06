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
#include <omp.h>

using namespace std;

const float INF = numeric_limits<float>::max();

struct Graph {
    int num_vertices;
    vector<vector<pair<int, float>>> adj_list; // adjacency list: [u] -> [(v, weight)]
    
    // SSSP tree data structures
    vector<float> distance;    // Dist array
    vector<int> parent;        // Parent array
    vector<bool> affected;     // Affected array
    vector<bool> affected_del; // Affected_Del array
    
    Graph(int n) : num_vertices(n), 
                  adj_list(n),
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

void process_changed_edges(Graph& graph, 
                         const vector<pair<pair<int, int>, float>>& insertions,
                         const vector<pair<int, int>>& deletions) {
    
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < graph.num_vertices; i++) {
            graph.affected[i] = false;
            graph.affected_del[i] = false;
        }

        // Process deletions in parallel
        #pragma omp for nowait
        for (size_t i = 0; i < deletions.size(); i++) {
            int u = deletions[i].first;
            int v = deletions[i].second;
            
            if (graph.parent[v] == u || graph.parent[u] == v) {
                int y = (graph.distance[u] > graph.distance[v]) ? u : v;
                #pragma omp critical
                {
                    graph.distance[y] = INF;
                    graph.parent[y] = -1;
                    graph.affected_del[y] = true;
                    graph.affected[y] = true;
                }
            }
        }

        // Process insertions in parallel
        #pragma omp for
        for (size_t i = 0; i < insertions.size(); i++) {
            int u = insertions[i].first.first;
            int v = insertions[i].first.second;
            float weight = insertions[i].second;
            
            int x = (graph.distance[u] < graph.distance[v]) ? u : v;
            int y = (x == u) ? v : u;
            
            if (graph.distance[y] > graph.distance[x] + weight) {
                #pragma omp critical
                {
                    graph.distance[y] = graph.distance[x] + weight;
                    graph.parent[y] = x;
                    graph.affected[y] = true;
                }
            }
        }
    }
}

void update_affected_vertices(Graph& graph) {
    bool changed = true;
    
    // Part 1: Process deletion-affected vertices
    while (changed) {
        changed = false;
        vector<int> current_affected;
        
        #pragma omp parallel for
        for (int v = 0; v < graph.num_vertices; v++) {
            if (graph.affected_del[v]) {
                #pragma omp critical
                current_affected.push_back(v);
            }
        }
        
        #pragma omp parallel for reduction(||:changed)
        for (size_t i = 0; i < current_affected.size(); i++) {
            int v = current_affected[i];
            graph.affected_del[v] = false;
            
            for (int c = 0; c < graph.num_vertices; c++) {
                if (graph.parent[c] == v) {
                    #pragma omp critical
                    {
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
        vector<int> current_affected;
        
        #pragma omp parallel for
        for (int v = 0; v < graph.num_vertices; v++) {
            if (graph.affected[v]) {
                #pragma omp critical
                current_affected.push_back(v);
            }
        }
        
        #pragma omp parallel for reduction(||:changed)
        for (size_t i = 0; i < current_affected.size(); i++) {
            int v = current_affected[i];
            graph.affected[v] = false;
            
            for (const auto& neighbor : graph.adj_list[v]) {
                int n = neighbor.first;
                float weight = neighbor.second;
                
                bool updated = false;
                #pragma omp critical
                {
                    if (graph.distance[n] > graph.distance[v] + weight) {
                        graph.distance[n] = graph.distance[v] + weight;
                        graph.parent[n] = v;
                        graph.affected[n] = true;
                        updated = true;
                    }
                    
                    if (graph.distance[v] > graph.distance[n] + weight) {
                        graph.distance[v] = graph.distance[n] + weight;
                        graph.parent[v] = n;
                        graph.affected[v] = true;
                        updated = true;
                    }
                }
                if (updated) changed = true;
            }
        }
    }
}

void initialize_sssp(Graph& graph, int source) {
    #pragma omp parallel for
    for (int i = 0; i < graph.num_vertices; i++) {
        graph.distance[i] = INF;
        graph.parent[i] = -1;
    }
    
    graph.distance[source] = 0;
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> pq;
    pq.push({0, source});
    
    while (!pq.empty()) {
        auto [dist_u, u] = pq.top();
        pq.pop();
        
        if (dist_u > graph.distance[u]) continue;
        
        #pragma omp parallel for
        for (size_t i = 0; i < graph.adj_list[u].size(); i++) {
            auto [v, weight] = graph.adj_list[u][i];
            bool updated = false;
            
            #pragma omp critical
            {
                if (graph.distance[v] > graph.distance[u] + weight) {
                    graph.distance[v] = graph.distance[u] + weight;
                    graph.parent[v] = u;
                    updated = true;
                }
            }
            
            if (updated) {
                #pragma omp critical
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
    double start_time = omp_get_wtime();
    process_changed_edges(graph, insertions, deletions);
    update_affected_vertices(graph);
    double end_time = omp_get_wtime();
    
    cout << "Update completed in " << end_time - start_time << " seconds." << endl;
}

// Function to print the shortest paths table with formatting
void print_sssp_table(const Graph& graph, int source) {
    cout << "\nCurrent shortest paths from source " << source << ":" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Vertex | Distance | Parent" << endl;
    cout << "----------------------------------------" << endl;
    
    // For small graphs (less than 15 vertices), show all vertices by default
    if (graph.num_vertices <= 15) {
        for (int v = 0; v < graph.num_vertices; v++) {
            cout << setw(6) << left << v << " | ";
            if (graph.distance[v] == INF)
                cout << setw(8) << left << "INF";
            else
                cout << setw(8) << left << graph.distance[v];
            cout << " | ";
            
            if (graph.parent[v] == -1)
                cout << "-";
            else
                cout << graph.parent[v];
                
            cout << endl;
        }
    } else {
        // For larger graphs, offer viewing options
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
                        cout << "-";
                    else
                        cout << graph.parent[v];
                        
                    cout << endl;
                }
                break;
            }
            case 2: {
                cout << "Enter vertices to show (comma-separated, e.g., 0,1,5,10): ";
                cin.ignore(); // Clear newline from buffer
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
                                cout << "-";
                            else
                                cout << graph.parent[v];
                                
                            cout << endl;
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
                        cout << "-";
                    else
                        cout << graph.parent[v];
                        
                    cout << endl;
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
                
                outfile << "Vertex,Distance,Parent" << endl;
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
                        
                    outfile << endl;
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
                        cout << "-";
                    else
                        cout << graph.parent[v];
                        
                    cout << endl;
                }
        }
    }
    
    cout << "----------------------------------------" << endl;
}

// Helper function to build the tree representation recursively
void build_tree_representation(const Graph& graph, int node, vector<string>& lines, string prefix = "") {
    // Find all children of this node
    vector<int> children;
    for (int v = 0; v < graph.num_vertices; v++) {
        if (graph.parent[v] == node && v != node) {
            children.push_back(v);
        }
    }
    
    // Sort children by distance
    sort(children.begin(), children.end(), [&graph](int a, int b) {
        return graph.distance[a] < graph.distance[b];
    });
    
    // Print this node
    if (node >= 0) { // Skip for the initial call
        string line = prefix + "└── " + to_string(node) + " (" + 
                     (graph.distance[node] == INF ? "INF" : to_string((int)graph.distance[node])) + ")";
        lines.push_back(line);
        prefix += "    ";
    }
    
    // Print children recursively
    for (size_t i = 0; i < children.size(); i++) {
        build_tree_representation(graph, children[i], lines, prefix);
    }
}

// Function to print the shortest path tree
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

int main() {
    // Set number of threads
    omp_set_num_threads(omp_get_max_threads());
    cout << "Running with " << omp_get_max_threads() << " threads" << endl;

    string filename;
    cout << "Enter graph file path (or press Enter to use default 'graph.txt'): ";
    getline(cin, filename);
    
    if (filename.empty()) {
        filename = "graph.txt"; // Default filename
    }
    
    cout << "Reading graph from " << filename << endl;
    double start = omp_get_wtime();
    Graph graph = read_mtx_graph(filename);
    double end = omp_get_wtime();
    cout << "Graph loaded with " << graph.num_vertices << " vertices in " 
         << end - start << " seconds" << endl;

    // Ask user for source vertex
    int source;
    cout << "\nEnter source vertex (0-" << graph.num_vertices-1 << "): ";
    cin >> source;
    
    // Validate input
    if (source < 0 || source >= graph.num_vertices) {
        cout << "Invalid source vertex. Using default source 0." << endl;
        source = 0;
    }

    cout << "\nComputing initial SSSP from vertex " << source << endl;
    start = omp_get_wtime();
    initialize_sssp(graph, source);
    end = omp_get_wtime();
    cout << "SSSP initialized in " << end - start << " seconds" << endl;

    // Print formatted results
    print_sssp_table(graph, source);
    print_shortest_path_tree(graph, source);

    // Optional: Process updates
    char apply_updates;
    cout << "\nApply graph updates? (y/n): ";
    cin >> apply_updates;
    
    if (apply_updates == 'y' || apply_updates == 'Y') {
        // Get user input for insertions and deletions
        vector<pair<pair<int, int>, float>> insertions;
        vector<pair<int, int>> deletions;
        
        char more = 'y';
        
        // Handle insertions
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
        
        // Handle deletions
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
        
        // Verification option
        char verify;
        cout << "\nVerify results by recomputing from scratch? (y/n): ";
        cin >> verify;
        
        if (verify == 'y' || verify == 'Y') {
            cout << "\nVerifying results..." << endl;
            start = omp_get_wtime();
            Graph recomputed_graph = read_mtx_graph(filename);
            
            // Apply changes to the graph structure
            for (const auto& ins : insertions) {
                int u = ins.first.first;
                int v = ins.first.second;
                float w = ins.second;
                recomputed_graph.adj_list[u].emplace_back(v, w);
                recomputed_graph.adj_list[v].emplace_back(u, w);
            }
            
            for (const auto& del : deletions) {
                int u = del.first;
                int v = del.second;
                auto& neighbors_u = recomputed_graph.adj_list[u];
                neighbors_u.erase(remove_if(neighbors_u.begin(), neighbors_u.end(),
                    [v](const pair<int, float>& e) { return e.first == v; }), neighbors_u.end());
                
                auto& neighbors_v = recomputed_graph.adj_list[v];
                neighbors_v.erase(remove_if(neighbors_v.begin(), neighbors_v.end(),
                    [u](const pair<int, float>& e) { return e.first == u; }), neighbors_v.end());
            }
            
            initialize_sssp(recomputed_graph, source);
            end = omp_get_wtime();
            cout << "Recomputation completed in " << end - start << " seconds" << endl;
            
            bool correct = true;
            int discrepancies = 0;
            const int max_discrepancies_to_show = 5;
            const int check_every = max(1, graph.num_vertices / 1000); // Check 0.1% of vertices
            
            #pragma omp parallel for reduction(+:discrepancies)
            for (int v = 0; v < graph.num_vertices; v += check_every) {
                if (fabs(graph.distance[v] - recomputed_graph.distance[v]) > 1e-6) {
                    #pragma omp critical
                    {
                        if (discrepancies < max_discrepancies_to_show) {
                            cout << "Discrepancy at vertex " << v << ": " 
                                 << graph.distance[v] << " (update) vs " 
                                 << recomputed_graph.distance[v] << " (recomputed)" << endl;
                        }
                        discrepancies++;
                        correct = false;
                    }
                }
            }
            
            if (correct) {
                cout << "Verification successful: All checked distances match!" << endl;
            } else {
                cout << "Found " << discrepancies << " discrepancies in sampled vertices" << endl;
                if (discrepancies > max_discrepancies_to_show) {
                    cout << "(Showing first " << max_discrepancies_to_show << ")" << endl;
                }
            }
        }
    }

    return 0;
}