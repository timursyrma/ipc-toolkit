// Hypothesis-generating measurements on real simulation scenes.
//
// H1 (distance-aware culling): of the EE candidates the production broad
//     phase emits (inflation = dhat/2, as NormalCollisions::build does), what
//     fraction is actually within dhat (i.e. contributes to the barrier)?
//     1/fraction = the headroom of a distance-based traversal.
// H2 (temporal coherence): how similar are the candidate sets of two
//     consecutive simulation frames (Jaccard), and how far do vertices move
//     relative to dhat? High overlap + small motion = warm-start/refit room.
#include <ipc/candidates/candidates.hpp>
#include <ipc/broad_phase/lbvh.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_triangle.hpp>
#ifdef IPC_TOOLKIT_WITH_CUDA
#include <ipc/broad_phase/gpu_lbvh.hpp>
#endif

#include <chrono>
#include <random>

#include <igl/read_triangle_mesh.h>
#include <igl/edges.h>

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace ipc;

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

static std::vector<EdgeEdgeCandidate> detect_ee(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const double inflation_radius)
{
    LBVH bp;
    bp.build(V, E, F, inflation_radius);
    std::vector<EdgeEdgeCandidate> ee;
    bp.detect_edge_edge_candidates(ee);
    return ee;
}

static std::vector<uint64_t> pair_keys(const std::vector<EdgeEdgeCandidate>& ee)
{
    std::vector<uint64_t> keys(ee.size());
    for (size_t i = 0; i < ee.size(); ++i) {
        const uint64_t a = std::min(ee[i].edge0_id, ee[i].edge1_id);
        const uint64_t b = std::max(ee[i].edge0_id, ee[i].edge1_id);
        keys[i] = (a << 32) | b;
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <tests-data-dir> [dhat_rel]\n", argv[0]);
        return 1;
    }
    const std::string data_dir = std::string(argv[1]) + "/";
    const double dhat_rel = argc > 2 ? std::atof(argv[2]) : 1e-3;

    printf(
        "dhat = %g * bbox_diag, inflation = dhat/2 (как в проде)\n\n",
        dhat_rel);

    for (const auto& scene : SCENES) {
        Eigen::MatrixXd V0, V1;
        Eigen::MatrixXi F0, F1, E;
        if (!igl::read_triangle_mesh(data_dir + scene.mesh_t0, V0, F0)
            || !igl::read_triangle_mesh(data_dir + scene.mesh_t1, V1, F1)) {
            fprintf(stderr, "failed to load %s\n", scene.name.c_str());
            continue;
        }
        igl::edges(F0, E);

        const double diag =
            (V0.colwise().maxCoeff() - V0.colwise().minCoeff()).norm();
        const double dhat = dhat_rel * diag;

        // --- H1: live fraction at t0 --------------------------------------
        const auto ee0 = detect_ee(V0, E, F0, 0.5 * dhat);

        const double dhat_sq = dhat * dhat;
        const size_t n_live = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, ee0.size()), size_t(0),
            [&](const tbb::blocked_range<size_t>& r, size_t acc) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    const auto& c = ee0[i];
                    const double d_sq = edge_edge_distance(
                        V0.row(E(c.edge0_id, 0)), V0.row(E(c.edge0_id, 1)),
                        V0.row(E(c.edge1_id, 0)), V0.row(E(c.edge1_id, 1)));
                    acc += (d_sq < dhat_sq) ? 1 : 0;
                }
                return acc;
            },
            std::plus<size_t>());

        // --- H2: temporal coherence t0 -> t1 ------------------------------
        const auto ee1 = detect_ee(V1, E, F1, 0.5 * dhat);
        const auto k0 = pair_keys(ee0);
        const auto k1 = pair_keys(ee1);
        std::vector<uint64_t> both;
        both.reserve(std::min(k0.size(), k1.size()));
        std::set_intersection(
            k0.begin(), k0.end(), k1.begin(), k1.end(),
            std::back_inserter(both));
        const size_t n_union = k0.size() + k1.size() - both.size();
        const double jaccard =
            n_union == 0 ? 1.0 : double(both.size()) / double(n_union);

        const double max_disp = (V1 - V0).rowwise().norm().maxCoeff();

        const double live_pct =
            ee0.empty() ? 0.0 : 100.0 * double(n_live) / double(ee0.size());
        printf(
            "%s: V=%d E=%d  dhat=%.3g\n", scene.name.c_str(), (int)V0.rows(),
            (int)E.rows(), dhat);
        printf(
            "  H1: EE candidates=%zu, live(<dhat)=%zu (%.2f%%) -> headroom %.0fx\n",
            ee0.size(), n_live, live_pct,
            n_live ? double(ee0.size()) / double(n_live) : 0.0);
        printf(
            "  H2: |t0 n t1|=%zu, Jaccard=%.3f, max_disp/dhat=%.2f\n",
            both.size(), jaccard, max_disp / dhat);

#ifdef IPC_TOOLKIT_WITH_CUDA
        // --- GPU exact dhat-culling vs plain GPU --------------------------
        using Clock = std::chrono::high_resolution_clock;
        auto run_gpu = [&](bool exact) {
            GPU_LBVH bp;
            bp.exact_ee_culling = exact;
            bp.build(V0, E, F0, 0.5 * dhat); // warm-up incl. CUDA init
            std::vector<EdgeEdgeCandidate> ee;
            bp.detect_edge_edge_candidates(ee);
            ee.clear();
            const auto t0 = Clock::now();
            constexpr int SAMPLES = 5;
            size_t n = 0;
            for (int i = 0; i < SAMPLES; ++i) {
                std::vector<EdgeEdgeCandidate> out;
                bp.detect_edge_edge_candidates(out);
                n = out.size();
            }
            const auto t1 = Clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count()
                / SAMPLES;
            return std::make_pair(n, ms);
        };
        const auto [n_plain, ms_plain] = run_gpu(false);
        const auto [n_exact, ms_exact] = run_gpu(true);
        printf(
            "  GPU EE: plain=%zu (%.2f ms) -> exact=%zu (%.2f ms)  сжатие "
            "%.1fx, live(CPU)=%zu%s\n",
            n_plain, ms_plain, n_exact, ms_exact,
            n_exact ? double(n_plain) / double(n_exact) : 0.0, n_live,
            n_exact >= n_live ? "" : "  LOST CANDIDATES!");

        // --- FV: same comparison -----------------------------------------
        auto run_gpu_fv = [&](bool exact) {
            GPU_LBVH bp;
            bp.exact_ee_culling = exact;
            bp.build(V0, E, F0, 0.5 * dhat);
            std::vector<FaceVertexCandidate> warm;
            bp.detect_face_vertex_candidates(warm);
            const auto t0 = Clock::now();
            constexpr int SAMPLES = 5;
            size_t n = 0;
            for (int i = 0; i < SAMPLES; ++i) {
                std::vector<FaceVertexCandidate> out;
                bp.detect_face_vertex_candidates(out);
                n = out.size();
            }
            const auto t1 = Clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count()
                / SAMPLES;
            return std::make_pair(n, ms);
        };
        const auto [fv_plain, fv_ms_plain] = run_gpu_fv(false);
        const auto [fv_exact, fv_ms_exact] = run_gpu_fv(true);

        // CPU-эталон живых FV пар
        std::vector<FaceVertexCandidate> fv0;
        {
            LBVH bp;
            bp.build(V0, E, F0, 0.5 * dhat);
            bp.detect_face_vertex_candidates(fv0);
        }
        const size_t fv_live = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, fv0.size()), size_t(0),
            [&](const tbb::blocked_range<size_t>& r, size_t acc) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    const auto& c = fv0[i];
                    const double d_sq = point_triangle_distance(
                        V0.row(c.vertex_id), V0.row(F0(c.face_id, 0)),
                        V0.row(F0(c.face_id, 1)), V0.row(F0(c.face_id, 2)));
                    acc += (d_sq < dhat_sq) ? 1 : 0;
                }
                return acc;
            },
            std::plus<size_t>());
        printf(
            "  GPU FV: plain=%zu (%.2f ms) -> exact=%zu (%.2f ms)  сжатие "
            "%.1fx, live(CPU)=%zu%s\n",
            fv_plain, fv_ms_plain, fv_exact, fv_ms_exact,
            fv_exact ? double(fv_plain) / double(fv_exact) : 0.0, fv_live,
            fv_exact >= fv_live ? "" : "  LOST CANDIDATES!");

        // --- VV/EV: validate the culled paths on small scenes -------------
        if (V0.rows() < 15000) {
            GPU_LBVH bp;
            bp.exact_ee_culling = true;
            bp.build(V0, E, F0, 0.5 * dhat);
            std::vector<VertexVertexCandidate> vv;
            bp.detect_vertex_vertex_candidates(vv);
            std::vector<EdgeVertexCandidate> ev;
            bp.detect_edge_vertex_candidates(ev);

            LBVH cpu;
            cpu.build(V0, E, F0, 0.5 * dhat);
            std::vector<VertexVertexCandidate> vv0;
            cpu.detect_vertex_vertex_candidates(vv0);
            std::vector<EdgeVertexCandidate> ev0;
            cpu.detect_edge_vertex_candidates(ev0);

            size_t vv_live = 0;
            for (const auto& c : vv0) {
                if ((V0.row(c.vertex0_id) - V0.row(c.vertex1_id)).squaredNorm()
                    < dhat_sq) {
                    vv_live++;
                }
            }
            size_t ev_live = 0;
            for (const auto& c : ev0) {
                const Eigen::Vector3d p = V0.row(c.vertex_id);
                const Eigen::Vector3d a = V0.row(E(c.edge_id, 0));
                const Eigen::Vector3d b = V0.row(E(c.edge_id, 1));
                const Eigen::Vector3d d = b - a;
                const double len_sq = d.squaredNorm();
                double t = len_sq > 0 ? (p - a).dot(d) / len_sq : 0.0;
                t = std::clamp(t, 0.0, 1.0);
                if ((p - (a + t * d)).squaredNorm() < dhat_sq) {
                    ev_live++;
                }
            }
            printf(
                "  GPU VV: plain=%zu -> exact=%zu, live(CPU)=%zu%s   "
                "EV: plain=%zu -> exact=%zu, live(CPU)=%zu%s\n",
                vv0.size(), vv.size(), vv_live,
                vv.size() >= vv_live ? "" : "  LOST!", ev0.size(), ev.size(),
                ev_live, ev.size() >= ev_live ? "" : "  LOST!");
        }

        // --- Fused broad+narrow barrier energy (zero materialization) -----
        {
            // CPU reference: the same clamped-log barrier summed over the
            // live pairs of the CPU candidate sets, all in double.
            auto barrier = [&](double d_sq) {
                if (d_sq >= dhat_sq || d_sq <= 0.0) {
                    return 0.0;
                }
                const double g = d_sq - dhat_sq;
                return -g * g * std::log(d_sq / dhat_sq);
            };
            double e_ref = 0.0;
            for (const auto& c : ee0) {
                e_ref += barrier(edge_edge_distance(
                    V0.row(E(c.edge0_id, 0)), V0.row(E(c.edge0_id, 1)),
                    V0.row(E(c.edge1_id, 0)), V0.row(E(c.edge1_id, 1))));
            }
            for (const auto& c : fv0) {
                e_ref += barrier(point_triangle_distance(
                    V0.row(c.vertex_id), V0.row(F0(c.face_id, 0)),
                    V0.row(F0(c.face_id, 1)), V0.row(F0(c.face_id, 2))));
            }

            GPU_LBVH bp;
            bp.exact_ee_culling = true;
            bp.build(V0, E, F0, 0.5 * dhat);
            double e_fused = bp.fused_barrier_potential(dhat); // warm-up

            constexpr int SAMPLES = 5;
            auto t0 = Clock::now();
            for (int i = 0; i < SAMPLES; ++i) {
                e_fused = bp.fused_barrier_potential(dhat);
            }
            auto t1 = Clock::now();
            const double fused_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count()
                / SAMPLES;

            // Pipeline baseline: culled detect (EE + FV) to host, then the
            // same barrier loop on the host over the returned candidates.
            t0 = Clock::now();
            double e_pipe = 0.0;
            for (int i = 0; i < SAMPLES; ++i) {
                std::vector<EdgeEdgeCandidate> ee;
                bp.detect_edge_edge_candidates(ee);
                std::vector<FaceVertexCandidate> fv;
                bp.detect_face_vertex_candidates(fv);
                e_pipe = 0.0;
                for (const auto& c : ee) {
                    e_pipe += barrier(edge_edge_distance(
                        V0.row(E(c.edge0_id, 0)), V0.row(E(c.edge0_id, 1)),
                        V0.row(E(c.edge1_id, 0)), V0.row(E(c.edge1_id, 1))));
                }
                for (const auto& c : fv) {
                    e_pipe += barrier(point_triangle_distance(
                        V0.row(c.vertex_id), V0.row(F0(c.face_id, 0)),
                        V0.row(F0(c.face_id, 1)), V0.row(F0(c.face_id, 2))));
                }
            }
            t1 = Clock::now();
            const double pipe_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count()
                / SAMPLES;

            const double rel = e_ref != 0.0
                ? std::abs(e_fused - e_ref) / std::abs(e_ref)
                : std::abs(e_fused);
            printf(
                "  FUSED: E=%.6e (ref %.6e, rel.diff=%.1e)  fused=%.2f ms "
                "vs pipeline=%.2f ms (%.1fx), bytes avoided=%zu\n",
                e_fused, e_ref, rel, fused_ms, pipe_ms,
                fused_ms > 0 ? pipe_ms / fused_ms : 0.0,
                16 * (n_exact + fv_exact));
        }

        // --- Warm start: certified reuse under small motion ---------------
        {
            const double mu = 0.5;
            GPU_LBVH bp;
            bp.exact_ee_culling = true;
            bp.warm_start_margin = mu;

            auto t0 = Clock::now();
            bp.build(V0, E, F0, 0.5 * dhat);
            std::vector<EdgeEdgeCandidate> ee_cold;
            bp.detect_edge_edge_candidates(ee_cold);
            std::vector<FaceVertexCandidate> fv_cold;
            bp.detect_face_vertex_candidates(fv_cold);
            auto t1 = Clock::now();
            const double cold_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Newton-подобное микродвижение: каждый сдвиг <= mu*dhat/4
            Eigen::MatrixXd Vp = V0;
            std::mt19937 rng(7);
            std::uniform_real_distribution<double> d(-1.0, 1.0);
            const double step = mu * dhat / 4 / std::sqrt(3.0);
            for (Eigen::Index i = 0; i < Vp.rows(); ++i) {
                for (int k = 0; k < 3; ++k) {
                    Vp(i, k) += step * d(rng);
                }
            }

            t0 = Clock::now();
            bp.build(Vp, E, F0, 0.5 * dhat);
            std::vector<EdgeEdgeCandidate> ee_warm;
            bp.detect_edge_edge_candidates(ee_warm);
            std::vector<FaceVertexCandidate> fv_warm;
            bp.detect_face_vertex_candidates(fv_warm);
            t1 = Clock::now();
            const double warm_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Сертификат: все живые EE пары на Vp обязаны быть в кэше
            std::vector<EdgeEdgeCandidate> eep;
            {
                LBVH bp2;
                bp2.build(Vp, E, F0, 0.5 * dhat);
                bp2.detect_edge_edge_candidates(eep);
            }
            auto warm_keys = pair_keys(ee_warm);
            size_t missing = 0;
            for (const auto& c : eep) {
                const double d_sq = edge_edge_distance(
                    Vp.row(E(c.edge0_id, 0)), Vp.row(E(c.edge0_id, 1)),
                    Vp.row(E(c.edge1_id, 0)), Vp.row(E(c.edge1_id, 1)));
                if (d_sq < dhat_sq) {
                    const uint64_t a = std::min(c.edge0_id, c.edge1_id);
                    const uint64_t b = std::max(c.edge0_id, c.edge1_id);
                    if (!std::binary_search(
                            warm_keys.begin(), warm_keys.end(),
                            (a << 32) | b)) {
                        missing++;
                    }
                }
            }
            printf(
                "  WARM: cold=%.2f ms -> reuse=%.2f ms (%.0fx), "
                "reused=%s, missing_live=%zu%s\n\n",
                cold_ms, warm_ms, warm_ms > 0 ? cold_ms / warm_ms : 0.0,
                bp.last_build_reused() ? "yes" : "NO!", missing,
                missing == 0 ? "" : "  CERTIFICATE VIOLATED!");
        }
#else
        printf("\n");
#endif
        fflush(stdout);
    }

    return 0;
}
