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

    using BroadPhase::build;

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
};

} // namespace ipc

#endif
