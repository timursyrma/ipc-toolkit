#include <ipc/ipc.hpp>
#include <ipc/candidates/candidates.hpp>
#include <ipc/broad_phase/broad_phase.hpp>
#include <ipc/broad_phase/lbvh.hpp>
#include <ipc/broad_phase/spatial_hash.hpp>
#include <ipc/broad_phase/sweep_and_prune.hpp>
#include <ipc/broad_phase/brute_force.hpp>
#ifdef IPC_TOOLKIT_WITH_CUDA
#include <ipc/broad_phase/gpu_lbvh.hpp>
#endif

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace ipc;
using Clock = std::chrono::high_resolution_clock;

static double ms(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct Scene {
    Eigen::MatrixXd V0, V1;
    Eigen::MatrixXi E, F;
    std::string name;
};

static void add_edge(std::set<std::pair<int,int>>& es, int a, int b) {
    es.insert({std::min(a,b), std::max(a,b)});
}

static Scene make_random_scene(int n_vertices, double density, unsigned seed) {
    Scene s;
    s.name = std::to_string(n_vertices) + " verts";

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pos(0.0, 1.0);

    s.V0.resize(n_vertices, 3);
    s.V1.resize(n_vertices, 3);
    for (int i = 0; i < n_vertices; i++) {
        s.V0(i, 0) = pos(rng);
        s.V0(i, 1) = pos(rng);
        s.V0(i, 2) = pos(rng);
        s.V1(i, 0) = s.V0(i, 0) + 0.001 * pos(rng);
        s.V1(i, 1) = s.V0(i, 1) + 0.001 * pos(rng);
        s.V1(i, 2) = s.V0(i, 2) + 0.001 * pos(rng);
    }

    int n_faces = std::max(1, int(n_vertices * density * 0.5));
    s.F.resize(n_faces, 3);
    std::set<std::pair<int,int>> edge_set;
    for (int i = 0; i < n_faces; i++) {
        int a = (i * 3) % n_vertices;
        int b = (i * 3 + 1) % n_vertices;
        int c = (i * 3 + 2) % n_vertices;
        s.F(i, 0) = a;
        s.F(i, 1) = b;
        s.F(i, 2) = c;
        add_edge(edge_set, a, b);
        add_edge(edge_set, b, c);
        add_edge(edge_set, c, a);
    }

    int extra_edges = int(n_vertices * density) - edge_set.size();
    for (int i = 0; i < std::max(0, extra_edges); i++) {
        int a = int(pos(rng) * n_vertices) % n_vertices;
        int b = (a + 1 + int(pos(rng) * 10)) % n_vertices;
        add_edge(edge_set, a, b);
    }

    s.E.resize(edge_set.size(), 2);
    int ei = 0;
    for (auto& e : edge_set) {
        s.E(ei, 0) = e.first;
        s.E(ei, 1) = e.second;
        ei++;
    }

    return s;
}

static Scene make_grid_scene(int grid_size, unsigned seed) {
    Scene s;
    s.name = "grid " + std::to_string(grid_size) + "^3";

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> jitter(-0.01, 0.01);

    int n = grid_size * grid_size * grid_size;
    s.V0.resize(n, 3);
    s.V1.resize(n, 3);
    int idx = 0;
    for (int x = 0; x < grid_size; x++) {
        for (int y = 0; y < grid_size; y++) {
            for (int z = 0; z < grid_size; z++) {
                s.V0(idx, 0) = double(x) / grid_size + jitter(rng);
                s.V0(idx, 1) = double(y) / grid_size + jitter(rng);
                s.V0(idx, 2) = double(z) / grid_size + jitter(rng);
                s.V1(idx, 0) = s.V0(idx, 0) + 0.001 * jitter(rng);
                s.V1(idx, 1) = s.V0(idx, 1) + 0.001 * jitter(rng);
                s.V1(idx, 2) = s.V0(idx, 2) + 0.001 * jitter(rng);
                idx++;
            }
        }
    }

    std::set<std::pair<int,int>> edge_set;
    std::vector<Eigen::Vector3i> faces;
    auto vidx = [grid_size](int x, int y, int z) {
        return x * grid_size * grid_size + y * grid_size + z;
    };
    for (int x = 0; x < grid_size; x++) {
        for (int y = 0; y < grid_size; y++) {
            for (int z = 0; z < grid_size; z++) {
                if (x + 1 < grid_size) add_edge(edge_set, vidx(x,y,z), vidx(x+1,y,z));
                if (y + 1 < grid_size) add_edge(edge_set, vidx(x,y,z), vidx(x,y+1,z));
                if (z + 1 < grid_size) add_edge(edge_set, vidx(x,y,z), vidx(x,y,z+1));
                if (x + 1 < grid_size && y + 1 < grid_size) {
                    int a = vidx(x,y,z), b = vidx(x+1,y,z), c = vidx(x,y+1,z), d = vidx(x+1,y+1,z);
                    faces.emplace_back(a, b, c);
                    faces.emplace_back(d, b, c);
                    add_edge(edge_set, b, c); // diagonal
                }
            }
        }
    }
    s.E.resize(edge_set.size(), 2);
    int ei = 0;
    for (auto& e : edge_set) { s.E(ei,0) = e.first; s.E(ei,1) = e.second; ei++; }
    s.F.resize(faces.size(), 3);
    for (int i = 0; i < int(faces.size()); i++) s.F.row(i) = faces[i];

    return s;
}

static void run_benchmark(const Scene& s, double inflation_radius) {
    CollisionMesh mesh = CollisionMesh::build_from_full_mesh(s.V0, s.E, s.F);
    Eigen::MatrixXd V0 = mesh.vertices(s.V0);
    Eigen::MatrixXd V1 = mesh.vertices(s.V1);

    printf("  Scene: %s | V=%d E=%d F=%d\n", s.name.c_str(),
           (int)V0.rows(), (int)s.E.rows(), (int)s.F.rows());

    auto run_bp = [&](const std::string& name, std::shared_ptr<BroadPhase> bp) {
        bp->can_vertices_collide = mesh.can_collide;

        auto t0 = Clock::now();
        bp->build(V0, V1, mesh.edges(), mesh.faces(), inflation_radius);
        auto t1 = Clock::now();

        Candidates c;
        bp->detect_collision_candidates(c);
        auto t2 = Clock::now();

        int total = c.size();
        printf("    %-20s build=%8.2fms  detect=%8.2fms  total=%8.2fms  candidates=%d\n",
               name.c_str(), ms(t0, t1), ms(t1, t2), ms(t0, t2), total);
    };

    run_bp("BruteForce", std::make_shared<BruteForce>());
    run_bp("SweepAndPrune", std::make_shared<SweepAndPrune>());
    run_bp("SpatialHash", std::make_shared<SpatialHash>());
    run_bp("LBVH (CPU)", std::make_shared<LBVH>());
#ifdef IPC_TOOLKIT_WITH_CUDA
    run_bp("GPU_LBVH", std::make_shared<GPU_LBVH>());
#endif
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== GPU_LBVH Benchmark ===\n\n");

    for (int g : {10, 20, 30, 50}) {        Scene s = make_grid_scene(g, 42);
        printf("[Grid Scene]\n");
        run_benchmark(s, 1e-3);
        printf("\n");
    }

    return 0;
}
