#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <climits>
#include <algorithm>
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
    for (auto& [node, _] : graph) dist[node] = INT_MAX;
    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [current_dist, u] = pq.top(); pq.pop();
        if (current_dist > dist[u]) continue;
        for (auto& edge : graph[u]) {
            int v = edge.to, weight = edge.weight;
            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
}

void mark_disconnected(unordered_map<int, vector<Edge>>& graph,
                       unordered_map<int, int>& dist,
                       unordered_map<int, int>& parent,
                       int start, unordered_set<int>& affected) {
    queue<int> q;
    q.push(start);
    affected.insert(start);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        dist[u] = INT_MAX;
        parent[u] = -1;
        for (auto& edge : graph[u]) {
            int v = edge.to;
            if (affected.find(v) == affected.end() && parent[v] == u) {
                affected.insert(v);
                q.push(v);
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
        // Relax neighbors
        if (dist[u] + w < dist[v]) {
            dist[v] = dist[u] + w;
            parent[v] = u;
            pq.push({dist[v], v});
        }
        if (dist[v] + w < dist[u]) {
            dist[u] = dist[v] + w;
            parent[u] = v;
            pq.push({dist[u], u});
        }
    } else if (op == "delete") {
        // Remove edge
        graph[u].erase(remove_if(graph[u].begin(), graph[u].end(),
                                 [v](Edge e) { return e.to == v; }), graph[u].end());
        graph[v].erase(remove_if(graph[v].begin(), graph[v].end(),
                                 [u](Edge e) { return e.to == u; }), graph[v].end());
        // Mark affected subtree starting from child side
        if (parent[v] == u) {
            mark_disconnected(graph, dist, parent, v, affected);
        } else if (parent[u] == v) {
            mark_disconnected(graph, dist, parent, u, affected);
        }
        // Add all affected nodes to queue for recomputation
        for (int node : affected) {
            for (auto& edge : graph[node]) {
                int neighbor = edge.to, weight = edge.weight;
                if (dist[neighbor] != INT_MAX && dist[node] > dist[neighbor] + weight) {
                    dist[node] = dist[neighbor] + weight;
                    parent[node] = neighbor;
                    pq.push({dist[node], node});
                }
            }
        }
    }

    while (!pq.empty()) {
        auto [current_dist, z] = pq.top(); pq.pop();
        for (auto& edge : graph[z]) {
            int n = edge.to, weight = edge.weight;
            if (dist[n] > dist[z] + weight) {
                dist[n] = dist[z] + weight;
                parent[n] = z;
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

    // Example insert edge
    int u = 1, v = 2, w = 5;
    string op = "insert"; // or "delete" or "insert"
    update_sssp(graph, dist, parent, u, v, w, op, source);

    cout << "Distances from source " << source << ":\n";
    for (auto& [node, d] : dist)
        cout << "Node " << node << ": " << (d == INT_MAX ? -1 : d) << endl;

    return 0;
}
