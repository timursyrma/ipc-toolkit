// Benchmark GPU_LBVH vs CPU LBVH on the real simulation scenes used by the
// upstream benchmark_lbvh_stages.cu, with the same protocol: dynamic build
// (t0, t1), inflation_radius = 0, detect = edge-edge only, warm-up then 10
// timed samples.
#include <ipc/ipc.hpp>
#include <ipc/candidates/candidates.hpp>
#include <ipc/broad_phase/broad_phase.hpp>
#include <ipc/broad_phase/lbvh.hpp>
#ifdef IPC_TOOLKIT_WITH_CUDA
#include <ipc/broad_phase/gpu_lbvh.hpp>
#endif

#include <igl/read_triangle_mesh.h>
#include <igl/edges.h>

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace ipc;
using Clock = std::chrono::high_resolution_clock;

static constexpr int SAMPLES = 10;
static constexpr double INFLATION_RADIUS = 0;

struct Scene {
    std::string name, mesh_t0, mesh_t1;
};

static const std::vector<Scene> SCENES = {
    { "Cloth-Funnel", "cloth-funnel/227.ply", "cloth-funnel/228.ply" },
    { "Armadillo-Rollers", "armadillo-rollers/326.ply",
      "armadillo-rollers/327.ply" },
    { "Rod-Twist", "rod-twist/3036.ply", "rod-twist/3037.ply" },
    { "Cloth-Ball", "cloth_ball92.ply", "cloth_ball93.ply" },
    { "N-Body-Simulation", "n-body-simulation/balls16_18.ply",
      "n-body-simulation/balls16_19.ply" },
    { "Puffer-Ball", "puffer-ball/20.ply", "puffer-ball/21.ply" },
};

template <typename F> static double time_calls(F&& f)
{
    f(); // warm up
    const auto t0 = Clock::now();
    for (int i = 0; i < SAMPLES; ++i) {
        f();
    }
    const auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count()
        / SAMPLES;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <tests-data-dir>\n", argv[0]);
        return 1;
    }
    const std::string data_dir = std::string(argv[1]) + "/";

    printf("samples=%d (среднее на вызов, ms)\n\n", SAMPLES);
    printf(
        "%-22s %7s %8s %8s %9s %9s\n", "Scene", "V", "cpu_bld", "gpu_bld",
        "cpu_det", "gpu_det");

    for (const auto& scene : SCENES) {
        Eigen::MatrixXd V0, V1;
        Eigen::MatrixXi F0, F1, E;
        if (!igl::read_triangle_mesh(data_dir + scene.mesh_t0, V0, F0)
            || !igl::read_triangle_mesh(data_dir + scene.mesh_t1, V1, F1)) {
            fprintf(stderr, "failed to load %s\n", scene.name.c_str());
            continue;
        }
        igl::edges(F0, E);

        LBVH cpu;
        const double cpu_build = time_calls([&]() {
            cpu.build(V0, V1, E, F0, INFLATION_RADIUS);
        });
        size_t n_cpu = 0;
        const double cpu_detect = time_calls([&]() {
            std::vector<EdgeEdgeCandidate> candidates;
            cpu.detect_edge_edge_candidates(candidates);
            n_cpu = candidates.size();
        });

#ifdef IPC_TOOLKIT_WITH_CUDA
        GPU_LBVH gpu;
        const double gpu_build = time_calls([&]() {
            gpu.build(V0, V1, E, F0, INFLATION_RADIUS);
        });
        size_t n_gpu = 0;
        const double gpu_detect = time_calls([&]() {
            std::vector<EdgeEdgeCandidate> candidates;
            gpu.detect_edge_edge_candidates(candidates);
            n_gpu = candidates.size();
        });
#else
        const double gpu_build = 0, gpu_detect = 0;
        const size_t n_gpu = 0;
#endif

        printf(
            "%-22s %7d %8.2f %8.2f %9.2f %9.2f  ee: cpu=%zu gpu=%zu%s\n",
            scene.name.c_str(), (int)V0.rows(), cpu_build, gpu_build,
            cpu_detect, gpu_detect, n_cpu, n_gpu,
            n_cpu == n_gpu ? "" : "  MISMATCH!");
        fflush(stdout);
    }

    return 0;
}
