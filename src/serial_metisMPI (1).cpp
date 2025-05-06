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
#include <mpi.h>   // MPI library for distributed computing

using namespace std;

const float INF = numeric_limits<float>::max();

// MPI tags
const int TAG_GRAPH_SIZE = 1;
const int TAG_ADJACENCY_LIST = 2;
const int TAG_PARTITION_DATA = 3;
const int TAG_DISTANCE = 4;
const int TAG_PARENT = 5;
const int TAG_UPDATE = 6;
const int TAG_AFFECTED = 7;
const int TAG_RESULT = 8;
const int TAG_TERMINATE = 9;

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

// Function to serialize adjacency list for MPI communication
void serialize_adj_list(const vector<vector<pair<int, float>>>& adj_list, 
                        vector<int>& vertices, 
                        vector<int>& edges, 
                        vector<float>& weights) {
    
    vertices.clear();
    edges.clear();
    weights.clear();
    
    vertices.push_back(0);
    for (const auto& neighbors : adj_list) {
        for (const auto& [v, w] : neighbors) {
            edges.push_back(v);
            weights.push_back(w);
        }
        vertices.push_back(edges.size());
    }
}

// Function to deserialize adjacency list received via MPI
void deserialize_adj_list(vector<vector<pair<int, float>>>& adj_list,
                         const vector<int>& vertices,
                         const vector<int>& edges,
                         const vector<float>& weights) {
    
    adj_list.resize(vertices.size() - 1);
    for (size_t i = 0; i < adj_list.size(); ++i) {
        adj_list[i].clear();
        for (int j = vertices[i]; j < vertices[i + 1]; ++j) {
            adj_list[i].emplace_back(edges[j], weights[j]);
        }
    }
}

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

void update_affected_vertices_parallel(Graph& graph, int rank, int size) {
    int vertices_per_proc = graph.num_vertices / size;
    int start_v = rank * vertices_per_proc;
    int end_v = (rank == size - 1) ? graph.num_vertices : (rank + 1) * vertices_per_proc;
    
    // Use char vectors for MPI communication
    vector<char> local_affected_char(graph.num_vertices, false);
    vector<float> local_distance = graph.distance;
    vector<int> local_parent = graph.parent;
    
    // Convert affected_del to char vector
    vector<char> affected_del_char(graph.affected_del.begin(), graph.affected_del.end());
    
    // Part 1: Process deletion-affected vertices
    bool local_changed = true, global_changed;
    
    while (true) {
        local_changed = false;
        
        for (int v = start_v; v < end_v; v++) {
            if (affected_del_char[v]) {
                affected_del_char[v] = false;
                
                for (int c = 0; c < graph.num_vertices; c++) {
                    if (graph.parent[c] == v) {
                        local_distance[c] = INF;
                        local_parent[c] = -1;
                        affected_del_char[c] = true;
                        local_affected_char[c] = true;
                        local_changed = true;
                    }
                }
            }
        }
        
        // Aggregate the changed status across all processes
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        
        // Share affected_del array with all processes using char vector
        MPI_Allreduce(MPI_IN_PLACE, affected_del_char.data(), graph.num_vertices, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        
        // Exit the loop if no changes were made by any process
        if (!global_changed) break;
    }
    
    // Update distance and parent
    MPI_Allreduce(local_distance.data(), graph.distance.data(), graph.num_vertices, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    
    // For parent array, we need a custom reduction that preserves parent-distance relationship
    for (int v = 0; v < graph.num_vertices; v++) {
        if (local_distance[v] < graph.distance[v]) {
            graph.distance[v] = local_distance[v];
            graph.parent[v] = local_parent[v];
        }
    }
    
    // Part 2: Update distances of affected vertices
    // Copy affected array to local_affected_char
    vector<char> affected_char(graph.affected.begin(), graph.affected.end());
    for (int i = 0; i < graph.num_vertices; i++) {
        if (affected_char[i]) local_affected_char[i] = true;
    }
    
    local_changed = true;
    while (true) {
        local_changed = false;
        
        for (int v = start_v; v < end_v; v++) {
            if (local_affected_char[v]) {
                local_affected_char[v] = false;
                
                for (const auto& neighbor : graph.adj_list[v]) {
                    int n = neighbor.first;
                    float weight = neighbor.second;
                    
                    if (local_distance[n] > local_distance[v] + weight) {
                        local_distance[n] = local_distance[v] + weight;
                        local_parent[n] = v;
                        local_affected_char[n] = true;
                        local_changed = true;
                    }
                    
                    if (local_distance[v] > local_distance[n] + weight) {
                        local_distance[v] = local_distance[n] + weight;
                        local_parent[v] = n;
                        local_affected_char[v] = true;
                        local_changed = true;
                    }
                }
            }
        }
        
        // Aggregate the changed status across all processes
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        
        // Share local_affected_char array with all processes
        MPI_Allreduce(MPI_IN_PLACE, local_affected_char.data(), graph.num_vertices, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        
        // Share local_distance and local_parent arrays
        vector<float> temp_distance(graph.num_vertices, INF);
        MPI_Allreduce(local_distance.data(), temp_distance.data(), graph.num_vertices, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        
        // Update parent based on minimum distance
        for (int v = 0; v < graph.num_vertices; v++) {
            if (local_distance[v] < temp_distance[v]) {
                temp_distance[v] = local_distance[v];
                graph.parent[v] = local_parent[v];
            }
        }
        
        local_distance = temp_distance;
        
        // Exit the loop if no changes were made by any process
        if (!global_changed) break;
    }
    
    // Update final distance and parent arrays
    graph.distance = local_distance;
    
    // Convert char vectors back to bool vectors
    for (int i = 0; i < graph.num_vertices; i++) {
        graph.affected[i] = affected_char[i];
        graph.affected_del[i] = affected_del_char[i];
    }
}
void initialize_sssp_parallel(Graph& graph, int source, int rank, int size) {
    fill(graph.distance.begin(), graph.distance.end(), INF);
    fill(graph.parent.begin(), graph.parent.end(), -1);
    
    if (rank == 0) {
        graph.distance[source] = 0;
    }
    
    // Broadcast initial distance array
    MPI_Bcast(graph.distance.data(), graph.num_vertices, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Determine the vertices this process will handle
    int vertices_per_proc = graph.num_vertices / size;
    int start_v = rank * vertices_per_proc;
    int end_v = (rank == size - 1) ? graph.num_vertices : (rank + 1) * vertices_per_proc;
    
    // Local priority queue for this process
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> pq;
    
    // Initial value for source vertex
    if (source >= start_v && source < end_v) {
        pq.push({0, source});
    }
    
    vector<float> local_distance = graph.distance;
    vector<int> local_parent = graph.parent;
    
    bool global_changed;
    do {
        bool local_changed = false;
        
        // Process the local priority queue
        while (!pq.empty()) {
            auto [dist_u, u] = pq.top();
            pq.pop();
            
            if (dist_u > local_distance[u]) continue;
            
            for (const auto& [v, weight] : graph.adj_list[u]) {
                if (local_distance[v] > local_distance[u] + weight) {
                    local_distance[v] = local_distance[u] + weight;
                    local_parent[v] = u;
                    if (v >= start_v && v < end_v) {
                        pq.push({local_distance[v], v});
                    }
                    local_changed = true;
                }
            }
        }
        
        // Share the changes with all processes
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        
        // Aggregate distances from all processes
        vector<float> global_distance(graph.num_vertices, INF);
        MPI_Allreduce(local_distance.data(), global_distance.data(), graph.num_vertices, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        
        // Update local distances and add newly changed vertices to queue
        for (int v = start_v; v < end_v; v++) {
            if (global_distance[v] < local_distance[v]) {
                local_distance[v] = global_distance[v];
                // We need to exchange parent info to maintain consistency
                // For simplicity, we'll do this in a separate step
                pq.push({local_distance[v], v});
            }
        }
        
        // Exchange parent information for vertices where distance changed
        for (int v = 0; v < graph.num_vertices; v++) {
            if (v >= start_v && v < end_v && local_distance[v] < global_distance[v]) {
                // This process has a better path to v
                struct {
                    int vertex;
                    int parent;
                } update_info = {v, local_parent[v]};
                
                for (int p = 0; p < size; p++) {
                    if (p != rank) {
                        MPI_Send(&update_info, sizeof(update_info), MPI_BYTE, p, TAG_PARENT, MPI_COMM_WORLD);
                    }
                }
            }
        }
        
        // Receive parent updates
        MPI_Status status;
        int flag;
        do {
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_PARENT, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                struct {
                    int vertex;
                    int parent;
                } update_info;
                
                MPI_Recv(&update_info, sizeof(update_info), MPI_BYTE, status.MPI_SOURCE, TAG_PARENT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                if (global_distance[update_info.vertex] < local_distance[update_info.vertex]) {
                    local_distance[update_info.vertex] = global_distance[update_info.vertex];
                    local_parent[update_info.vertex] = update_info.parent;
                }
            }
        } while (flag);
        
    } while (global_changed);
    
    // Update final distance and parent arrays
    graph.distance = local_distance;
    graph.parent = local_parent;
    
    // Make sure all processes have the same final arrays
    MPI_Bcast(graph.distance.data(), graph.num_vertices, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph.parent.data(), graph.num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
}

void update_sssp_parallel(Graph& graph, 
                         const vector<pair<pair<int, int>, float>>& insertions,
                         const vector<pair<int, int>>& deletions,
                         int rank, int size) {
    if (rank == 0) {
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
    }
    
    // Broadcast insertions and deletions to all processes
    int insertion_count = insertions.size();
    int deletion_count = deletions.size();
    
    MPI_Bcast(&insertion_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&deletion_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Master process processes the changes initially
    if (rank == 0) {
        process_changed_edges(graph, insertions, deletions);
    }
    
    // Create temporary char arrays for MPI communication
    vector<char> affected_char(graph.num_vertices);
    vector<char> affected_del_char(graph.num_vertices);
    
    // Convert bool vectors to char arrays on rank 0
    if (rank == 0) {
        for (int i = 0; i < graph.num_vertices; i++) {
            affected_char[i] = graph.affected[i];
            affected_del_char[i] = graph.affected_del[i];
        }
    }
    
    // Broadcast affected arrays to all processes using char arrays
    MPI_Bcast(affected_char.data(), graph.num_vertices, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(affected_del_char.data(), graph.num_vertices, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // Convert char arrays back to bool vectors on all processes
    for (int i = 0; i < graph.num_vertices; i++) {
        graph.affected[i] = affected_char[i];
        graph.affected_del[i] = affected_del_char[i];
    }
    
    // Each process updates its vertices in parallel
    update_affected_vertices_parallel(graph, rank, size);
    
    if (rank == 0) {
        cout << "Update completed." << endl;
    }
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

// Helper function to synchronize graph updates after edge insertions/deletions
void synchronize_graph_updates(Graph& graph, 
                              const vector<pair<pair<int, int>, float>>& insertions,
                              const vector<pair<int, int>>& deletions, 
                              int rank, int size) {
    // First, apply the edge changes to the local graph copy
    if (rank == 0) {
        // Process insertions
        for (const auto& edge : insertions) {
            int u = edge.first.first;
            int v = edge.first.second;
            float weight = edge.second;
            
            // Add the new edge to adjacency list of u
            bool exists = false;
            for (auto& neighbor : graph.adj_list[u]) {
                if (neighbor.first == v) {
                    neighbor.second = weight; // Update weight if edge already exists
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                graph.adj_list[u].emplace_back(v, weight);
            }
            
            // Add the new edge to adjacency list of v (undirected graph)
            exists = false;
            for (auto& neighbor : graph.adj_list[v]) {
                if (neighbor.first == u) {
                    neighbor.second = weight;
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                graph.adj_list[v].emplace_back(u, weight);
            }
        }
        
        // Process deletions
        for (const auto& edge : deletions) {
            int u = edge.first;
            int v = edge.second;
            
            // Remove edge from u's adjacency list
            auto& u_list = graph.adj_list[u];
            u_list.erase(
                remove_if(u_list.begin(), u_list.end(), 
                          [v](const pair<int, float>& e) { return e.first == v; }),
                u_list.end()
            );
            
            // Remove edge from v's adjacency list
            auto& v_list = graph.adj_list[v];
            v_list.erase(
                remove_if(v_list.begin(), v_list.end(), 
                          [u](const pair<int, float>& e) { return e.first == u; }),
                v_list.end()
            );
        }
    }
    
    // Now broadcast the updated graph structure to all processes
    for (int i = 0; i < graph.num_vertices; i++) {
        int adj_size = (rank == 0) ? graph.adj_list[i].size() : 0;
        MPI_Bcast(&adj_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        vector<pair<int, float>> adj_data;
        if (rank == 0) {
            adj_data = graph.adj_list[i];
        } else {
            adj_data.resize(adj_size);
        }
        
        // Serialize adjacency data for broadcasting
        vector<int> neighbors(adj_size);
        vector<float> weights(adj_size);
        
        if (rank == 0) {
            for (int j = 0; j < adj_size; j++) {
                neighbors[j] = adj_data[j].first;
                weights[j] = adj_data[j].second;
            }
        }
        
        MPI_Bcast(neighbors.data(), adj_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(weights.data(), adj_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            graph.adj_list[i].clear();
            for (int j = 0; j < adj_size; j++) {
                graph.adj_list[i].emplace_back(neighbors[j], weights[j]);
            }
        }
    }
}

// Helper function to distribute the graph across MPI processes
void distribute_graph(Graph& graph, int rank, int size) {
    if (rank == 0) {
        // Master sends graph data to all worker processes
        int num_vertices = graph.num_vertices;
        
        for (int dest = 1; dest < size; dest++) {
            // Send number of vertices
            MPI_Send(&num_vertices, 1, MPI_INT, dest, TAG_GRAPH_SIZE, MPI_COMM_WORLD);
            
            // Serialize and send adjacency list
            vector<int> vertices, edges;
            vector<float> weights;
            serialize_adj_list(graph.adj_list, vertices, edges, weights);
            
            int v_size = vertices.size();
            int e_size = edges.size();
            
            MPI_Send(&v_size, 1, MPI_INT, dest, TAG_ADJACENCY_LIST, MPI_COMM_WORLD);
            MPI_Send(&e_size, 1, MPI_INT, dest, TAG_ADJACENCY_LIST, MPI_COMM_WORLD);
            MPI_Send(vertices.data(), v_size, MPI_INT, dest, TAG_ADJACENCY_LIST, MPI_COMM_WORLD);
            MPI_Send(edges.data(), e_size, MPI_INT, dest, TAG_ADJACENCY_LIST, MPI_COMM_WORLD);
            MPI_Send(weights.data(), e_size, MPI_FLOAT, dest, TAG_ADJACENCY_LIST, MPI_COMM_WORLD);
            
            // Send partition data
            MPI_Send(graph.partitions.data(), graph.num_vertices, MPI_INT, dest, TAG_PARTITION_DATA, MPI_COMM_WORLD);
        }
    } else {
        // Worker processes receive graph data from master
        int num_vertices;
        MPI_Recv(&num_vertices, 1, MPI_INT, 0, TAG_GRAPH_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Initialize graph with received size
        graph = Graph(num_vertices);
        
        // Receive and deserialize adjacency list
        int v_size, e_size;
        MPI_Recv(&v_size, 1, MPI_INT, 0, TAG_ADJACENCY_LIST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&e_size, 1, MPI_INT, 0, TAG_ADJACENCY_LIST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        vector<int> vertices(v_size), edges(e_size);
        vector<float> weights(e_size);
        
        MPI_Recv(vertices.data(), v_size, MPI_INT, 0, TAG_ADJACENCY_LIST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(edges.data(), e_size, MPI_INT, 0, TAG_ADJACENCY_LIST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(weights.data(), e_size, MPI_FLOAT, 0, TAG_ADJACENCY_LIST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        deserialize_adj_list(graph.adj_list, vertices, edges, weights);
        
        // Receive partition data
        MPI_Recv(graph.partitions.data(), graph.num_vertices, MPI_INT, 0, TAG_PARTITION_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Only rank 0 handles file I/O
    Graph graph(0);
    if (rank == 0) {
        string filename = "graph.txt"; // Default filename
        cout << "Reading graph from " << filename << endl;
        graph = read_mtx_graph(filename);
        cout << "Graph loaded with " << graph.num_vertices << " vertices" << endl;

        // Partition the graph
        partition_graph(graph, size);
    }

    // Broadcast graph size to all processes
    int num_vertices;
    if (rank == 0) {
        num_vertices = graph.num_vertices;
    }
    MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Initialize local graph structure
    Graph local_graph(num_vertices);

    // [Add code to distribute graph data to all processes...]

    // Get source vertex (only rank 0 prompts)
    int source = 0;
    if (rank == 0) {
        cout << "\nEnter source vertex (0-" << num_vertices-1 << "): ";
        cin >> source;
    }
    MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute SSSP
    initialize_sssp_parallel(local_graph, source, rank, size);

    // [Add code to handle updates if needed...]

    // Print results (only rank 0)
    if (rank == 0) {
        print_sssp_table(local_graph, source);
    }

    MPI_Finalize();
    return 0;
}
