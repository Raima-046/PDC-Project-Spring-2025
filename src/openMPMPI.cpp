#include <iostream>
#include <vector>
#include <limits>
#include <mpi.h>
#include <omp.h>
#include <metis.h>

const float INF = std::numeric_limits<float>::max();

struct Graph {
    std::vector<float> distance;
    std::vector<int> parent;
    std::vector<char> affected;      // Changed from bool to char for MPI
    std::vector<char> affected_del;  // Changed from bool to char for MPI
    std::vector<std::vector<std::pair<int, float>>> adj_list;
    int num_vertices;
    std::vector<int> part;
};

// Forward declaration
void initialize_graph(Graph &graph, int n);

Graph read_mtx_graph(const std::string &filename) {
    Graph graph;
    graph.num_vertices = 100; // Example size
    initialize_graph(graph, graph.num_vertices); // Now properly declared
    return graph;
}

void initialize_graph(Graph &graph, int n) {
    graph.distance.assign(n, INF);
    graph.parent.assign(n, -1);
    graph.affected.assign(n, 0);    // 0 instead of false
    graph.affected_del.assign(n, 0); // 0 instead of false
    graph.adj_list.resize(n);
    graph.part.assign(n, 0);
}




// Partition the graph using METIS (simplified)
void partition_with_metis(Graph &graph, int nparts) {
    idx_t nvtxs = graph.num_vertices;
    idx_t ncon = 1;
    idx_t *xadj = new idx_t[nvtxs + 1];
    idx_t *adjncy = new idx_t[graph.adj_list.size() * 2];  // Approximate
    idx_t *part = new idx_t[nvtxs];
    idx_t objval;

    // (In a real implementation, fill xadj and adjncy)
    METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, NULL, &nparts, NULL, NULL, NULL, &objval, part);

    for (int i = 0; i < nvtxs; ++i)
        graph.part[i] = part[i];

    delete[] xadj;
    delete[] adjncy;
    delete[] part;
}

// Print SSSP results (simplified)
void print_sssp_table(const Graph &graph, int source) {
    std::cout << "Vertex\tDistance\tParent\n";
    for (int i = 0; i < graph.num_vertices; ++i) {
        std::cout << i << "\t" << graph.distance[i] << "\t" << graph.parent[i] << "\n";
    }
}

// Main parallel SSSP update

void update_affected_vertices_parallel(Graph &graph, int rank, int size) {
    int n = graph.num_vertices;
    int start_v = rank * n / size;
    int end_v = (rank + 1) * n / size;

    std::vector<char> affected_del_char(n, 0);
    std::vector<char> local_affected_char(n, 0);
    std::vector<float> local_distance = graph.distance;
    std::vector<int> local_parent = graph.parent;

    if (rank == 0) {
        for (int i = 0; i < n; ++i)
            affected_del_char[i] = graph.affected_del[i];
    }

    MPI_Bcast(affected_del_char.data(), n, MPI_CHAR, 0, MPI_COMM_WORLD);

    while (true) {
        bool local_changed = false;
        #pragma omp parallel
        {
            bool thread_changed = false;
            #pragma omp for schedule(dynamic, 64)
            for (int v = start_v; v < end_v; ++v) {
                if (affected_del_char[v]) {
                    for (auto &nbr : graph.adj_list[v]) {
                        int c = nbr.first;
                        if (graph.parent[c] == v) {
                            local_distance[c] = INF;
                            local_parent[c] = -1;
                            thread_changed = true;
                        }
                    }
                }
            }
            #pragma omp atomic
            local_changed |= thread_changed;
        }

        bool global_changed;
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, affected_del_char.data(), n, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        if (!global_changed) break;
    }
    // REMOVED THE EXTRA BRACE HERE

    MPI_Allreduce(local_distance.data(), graph.distance.data(), n, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);

    for (int v = 0; v < n; ++v) {
        if (local_distance[v] < graph.distance[v]) {
            graph.distance[v] = local_distance[v];
            graph.parent[v] = local_parent[v];
        }
    }

    for (int i = 0; i < n; ++i)
        local_affected_char[i] = graph.affected[i];

    while (true) {
        bool local_changed = false;
        #pragma omp parallel
        {
            bool thread_changed = false;
            #pragma omp for schedule(dynamic, 64)
            for (int v = start_v; v < end_v; ++v) {
                if (local_affected_char[v]) {
                    local_affected_char[v] = false;
                    for (auto &nbr : graph.adj_list[v]) {
                        int u = nbr.first;
                        float wgt = nbr.second;
                        float nd = graph.distance[v] + wgt;
                        if (graph.distance[u] > nd) {
                            graph.distance[u] = nd;
                            graph.parent[u] = v;
                            local_affected_char[u] = true;
                            thread_changed = true;
                        }
                    }
                }
            }
            #pragma omp atomic
            local_changed |= thread_changed;
        }

        bool global_changed;
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, graph.affected_del.data(), n, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        if (!global_changed) break;
    }

    MPI_Allreduce(MPI_IN_PLACE, graph.distance.data(), n, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Read and partition the graph
    Graph graph = read_mtx_graph("graph.mtx");
    if (rank == 0) {
        partition_with_metis(graph, size);
    }
    MPI_Bcast(graph.part.data(), graph.num_vertices, MPI_INT, 0, MPI_COMM_WORLD);

    // 2. Initialize SSSP
    int source = 0;
    if (rank == 0) graph.distance[source] = 0;
    MPI_Bcast(graph.distance.data(), graph.num_vertices, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph.parent.data(), graph.num_vertices, MPI_INT, 0, MPI_COMM_WORLD);

    // 3. Run parallel SSSP updates
    update_affected_vertices_parallel(graph, rank, size);

    // 4. Print results (rank 0 only)
    if (rank == 0) {
        print_sssp_table(graph, source);
    }

    MPI_Finalize();
    return 0;
}
