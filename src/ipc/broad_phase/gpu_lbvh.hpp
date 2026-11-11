#pragma once

#include <ipc/config.hpp>

#ifdef IPC_TOOLKIT_WITH_CUDA

#include <ipc/broad_phase/broad_phase.hpp>

namespace ipc {

class GPU_LBVH : public BroadPhase {
public:
    GPU_LBVH();
    ~GPU_LBVH();

    std::string name() const override { return "GPU_LBVH"; }

    /// @brief Experimental: exact dhat-culling for edge-edge candidates.
    ///
    /// When enabled and the broad phase is built with the static build()
    /// (vertices, edges, faces, inflation_radius > 0), the EE traversal
    /// replaces the AABB leaf test with the exact segment-segment distance
    /// test against dhat = 2 * inflation_radius (the production convention:
    /// NormalCollisions::build passes inflation_radius = dhat / 2). Only
    /// pairs within dhat are emitted, so the EE candidate set shrinks to the
    /// pairs the barrier actually needs. The threshold carries a small
    /// relative margin for the float cast, so the set stays a superset of
    /// the exact one. Ignored by the dynamic (t0, t1) build.
    bool exact_ee_culling = false;

    /// @brief Experimental: certified warm-start margin mu (0 disables).
    ///
    /// Requires exact_ee_culling. When mu > 0, the static build() widens the
    /// culling threshold and the tree inflation to (1 + mu) * dhat. A later
    /// build() with the same topology and inflation measures the maximum
    /// vertex displacement delta since the cached build; if
    /// 2 * delta <= mu * dhat, every pair within dhat now was within
    /// (1 + mu) * dhat then, so the cached candidate sets are provable
    /// supersets of the live sets and both the tree build and the traversal
    /// are skipped entirely. Covers the distance-typed candidates
    /// (VV/EV/EE/FV, i.e. the barrier pipeline); EF/FF detected during a
    /// reused step run on the cached trees and are not certified.
    double warm_start_margin = 0;

    /// @brief Whether the last static build() reused the cached candidates.
    bool last_build_reused() const { return m_reuse; }

    using BroadPhase::build;

    void build(
        Eigen::ConstRef<Eigen::MatrixXd> vertices,
        Eigen::ConstRef<Eigen::MatrixXi> edges,
        Eigen::ConstRef<Eigen::MatrixXi> faces,
        const double inflation_radius = 0) override;

    void clear() override;

    void detect_vertex_vertex_candidates(
        std::vector<VertexVertexCandidate>& candidates) const override;
    void detect_edge_vertex_candidates(
        std::vector<EdgeVertexCandidate>& candidates) const override;
    void detect_edge_edge_candidates(
        std::vector<EdgeEdgeCandidate>& candidates) const override;
    void detect_face_vertex_candidates(
        std::vector<FaceVertexCandidate>& candidates) const override;
    void detect_edge_face_candidates(
        std::vector<EdgeFaceCandidate>& candidates) const override;
    void detect_face_face_candidates(
        std::vector<FaceFaceCandidate>& candidates) const override;

protected:
    void build(
        Eigen::ConstRef<Eigen::MatrixXi> edges,
        Eigen::ConstRef<Eigen::MatrixXi> faces) override;

    bool can_edge_vertex_collide(size_t ei, size_t vi) const override;
    bool can_edges_collide(size_t eai, size_t ebi) const override;
    bool can_face_vertex_collide(size_t fi, size_t vi) const override;
    bool can_edge_face_collide(size_t ei, size_t fi) const override;
    bool can_faces_collide(size_t fai, size_t fbi) const override;

    struct Impl;
    std::unique_ptr<Impl> impl;

private:
    // Staged by the public static build() and consumed by the protected
    // build(). Class members (not Impl) because BroadPhase::build() calls
    // clear(), which recreates the Impl, between the two.
    double m_staged_ee_cull_sq = -1;
    std::vector<float> m_staged_verts; ///< xyz triples (z = 0 in 2D).

    // Warm-start state (see warm_start_margin). The cached candidate sets
    // are filled lazily by the detect_*() calls after a caching build, so
    // they are mutable; m_reuse is only flipped by build()/clear().
    bool m_reuse = false;
    Eigen::MatrixXd m_cached_vertices;
    Eigen::Index m_cached_num_edges = -1;
    Eigen::Index m_cached_num_faces = -1;
    double m_cached_inflation = -1;
    mutable std::vector<VertexVertexCandidate> m_cached_vv;
    mutable std::vector<EdgeVertexCandidate> m_cached_ev;
    mutable std::vector<EdgeEdgeCandidate> m_cached_ee;
    mutable std::vector<FaceVertexCandidate> m_cached_fv;
    mutable bool m_has_vv = false, m_has_ev = false, m_has_ee = false,
                 m_has_fv = false;
};

} // namespace ipc

#endif
