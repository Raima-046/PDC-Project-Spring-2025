#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <climits>
#include <algorithm>
#include <omp.h> // Include OpenMP header
using namespace std;

typedef pair<int, int> pii;

struct Edge {
    int to, weight;
};

void read_graph(string filename, unordered_map<int, vector<Edge>>& graph) {
    ifstream file(filename);
    int u, v, w;
    while (file >> u >> v >> w) {
        graph[u].push_back({v, w});
        graph[v].push_back({u, w});
    }
}

void dijkstra(unordered_map<int, vector<Edge>>& graph, int source,
              unordered_map<int, int>& dist, unordered_map<int, int>& parent) {
    priority_queue<pii, vector<pii>, greater<pii>> pq;
    
    // Copy keys to a vector
    vector<int> keys;
    for (const auto& kv : graph) {
        keys.push_back(kv.first);
    }

    // Parallel initialization of dist and parent
    #pragma omp parallel for
    for (int i = 0; i < keys.size(); ++i) {
        int key = keys[i];
        dist[key] = INT_MAX;
        parent[key] = -1;
    }

    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [current_dist, u] = pq.top(); pq.pop();
        if (current_dist > dist[u]) continue;
        
        // Parallel relaxation over neighbors
        #pragma omp parallel for
        for (int i = 0; i < graph[u].size(); i++) {
            int v = graph[u][i].to;
            int weight = graph[u][i].weight;
            bool updated = false;

            #pragma omp critical
            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                parent[v] = u;
                updated = true;
            }
            if (updated) {
                #pragma omp critical
                pq.push({dist[v], v});
            }
        }
    }
}


void update_sssp(unordered_map<int, vector<Edge>>& graph, unordered_map<int, int>& dist,
                 unordered_map<int, int>& parent, int u, int v, int w, string op, int source) {
    priority_queue<pii, vector<pii>, greater<pii>> pq;
    unordered_set<int> affected;

    if (op == "insert") {
        graph[u].push_back({v, w});
        graph[v].push_back({u, w});
        
        // Relax in parallel
        #pragma omp parallel sections
        {
            #pragma omp section
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                #pragma omp critical
                pq.push({dist[v], v});
            }
            #pragma omp section
            if (dist[v] + w < dist[u]) {
                dist[u] = dist[v] + w;
                parent[u] = v;
                #pragma omp critical
                pq.push({dist[u], u});
            }
        }
    }
    else if (op == "delete") {
        graph[u].erase(remove_if(graph[u].begin(), graph[u].end(),
                                 [v](Edge e) { return e.to == v; }), graph[u].end());
        graph[v].erase(remove_if(graph[v].begin(), graph[v].end(),
                                 [u](Edge e) { return e.to == u; }), graph[v].end());
        if (parent[v] == u) {
            dist[v] = INT_MAX;
            parent[v] = -1;
            pq.push({dist[v], v});
        }
        if (parent[u] == v) {
            dist[u] = INT_MAX;
            parent[u] = -1;
            pq.push({dist[u], u});
        }
    }

    while (!pq.empty()) {
        auto [current_dist, z] = pq.top(); pq.pop();
        
        // Parallel relaxation over neighbors
        #pragma omp parallel for
        for (int i = 0; i < graph[z].size(); i++) {
            int n = graph[z][i].to;
            int weight = graph[z][i].weight;
            bool updated = false;

            #pragma omp critical
            if (dist[n] > dist[z] + weight) {
                dist[n] = dist[z] + weight;
                parent[n] = z;
                updated = true;
            }
            if (updated) {
                #pragma omp critical
                pq.push({dist[n], n});
            }
        }
    }
}

int main() {
    unordered_map<int, vector<Edge>> graph;
    unordered_map<int, int> dist, parent;
    string filename = "/mnt/c/Users/Jawairia/Documents/PDC Project/data/graph.txt";
    read_graph(filename, graph);

    cout << "Graph loaded with " << graph.size() << " nodes.\n";

    int source = 0;
    if (graph.find(source) == graph.end()) {
        cout << "Source node " << source << " not found in graph! Using first available node.\n";
        source = graph.begin()->first;
    }

    dijkstra(graph, source, dist, parent);

    int u = 1, v = 2, w = 5;
    string op = "insert"; // or "delete"
    update_sssp(graph, dist, parent, u, v, w, op, source);

    cout << "Distances from source " << source << ":\n";
    for (auto& [node, d] : dist)
        cout << "Node " << node << ": " << (d == INT_MAX ? -1 : d) << endl;

    return 0;
}
