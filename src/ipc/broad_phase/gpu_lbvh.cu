#include "gpu_lbvh.hpp"

#ifdef IPC_TOOLKIT_WITH_CUDA

#include <ipc/broad_phase/aabb.hpp>
#include <ipc/candidates/vertex_vertex.hpp>
#include <ipc/candidates/edge_vertex.hpp>
#include <ipc/candidates/edge_edge.hpp>
#include <ipc/candidates/face_vertex.hpp>
#include <ipc/candidates/edge_face.hpp>
#include <ipc/candidates/face_face.hpp>
#include <ipc/utils/profiler.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/gather.h>

#include <algorithm>
#include <vector>

namespace ipc {

namespace {

struct alignas(16) GpuNode {
    float aabb_min_x, aabb_min_y, aabb_min_z;
    float aabb_max_x, aabb_max_y, aabb_max_z;
    int32_t left;
    int32_t right;
    int32_t primitive_id;
    int32_t is_inner_marker;
};

struct CandidatePair {
    int32_t id_a;
    int32_t id_b;
};

__device__ __host__ inline bool gpu_node_is_inner(const GpuNode& n) { return n.is_inner_marker != 0; }
__device__ __host__ inline bool gpu_node_is_leaf(const GpuNode& n) { return n.is_inner_marker == 0; }

__device__ __host__ inline bool gpu_nodes_intersect(const GpuNode& a, const GpuNode& b)
{
    return a.aabb_min_x <= b.aabb_max_x && b.aabb_min_x <= a.aabb_max_x
        && a.aabb_min_y <= b.aabb_max_y && b.aabb_min_y <= a.aabb_max_y
        && a.aabb_min_z <= b.aabb_max_z && b.aabb_min_z <= a.aabb_max_z;
}

__device__ inline uint64_t expand_bits_2(uint64_t v)
{
    v = (v | v << 32) & 0x1F00000000FFFF;
    v = (v | v << 16) & 0x1F0000FF0000FF;
    v = (v | v << 8) & 0x100F00F00F00F00F;
    v = (v | v << 4) & 0x10C30C30C30C30C3;
    v = (v | v << 2) & 0x1249249249249249;
    return v;
}

__device__ inline uint64_t expand_bits_1(uint64_t v)
{
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF;
    v = (v | (v << 8)) & 0x00FF00FF00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F;
    v = (v | (v << 2)) & 0x3333333333333333;
    v = (v | (v << 1)) & 0x5555555555555555;
    return v;
}

__device__ inline uint64_t gpu_morton_3d(double x, double y, double z)
{
    constexpr double scale = (1ULL << 21);
    x = fmin(fmax(x * scale, 0.0), scale - 1);
    y = fmin(fmax(y * scale, 0.0), scale - 1);
    z = fmin(fmax(z * scale, 0.0), scale - 1);
    return (expand_bits_2(uint64_t(x)) << 2) | (expand_bits_2(uint64_t(y)) << 1) | expand_bits_2(uint64_t(z));
}

__device__ inline uint64_t gpu_morton_2d(double x, double y)
{
    constexpr double scale = (1ULL << 32);
    x = fmin(fmax(x * scale, 0.0), scale - 1);
    y = fmin(fmax(y * scale, 0.0), scale - 1);
    return (expand_bits_1(uint64_t(x)) << 1) | expand_bits_1(uint64_t(y));
}

__global__ void compute_morton_codes_kernel(
    const int num_primitives,
    const double mesh_min_x, const double mesh_min_y, const double mesh_min_z,
    const double mesh_width_x, const double mesh_width_y, const double mesh_width_z,
    const float* aabb_mins_x, const float* aabb_mins_y, const float* aabb_mins_z,
    const float* aabb_maxs_x, const float* aabb_maxs_y, const float* aabb_maxs_z,
    uint64_t* morton_codes,
    int32_t* box_ids,
    const uint8_t dim)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_primitives)
        return;

    double cx = 0.5 * (double(aabb_mins_x[i]) + double(aabb_maxs_x[i]));
    double cy = 0.5 * (double(aabb_mins_y[i]) + double(aabb_maxs_y[i]));
    double cz = 0.5 * (double(aabb_mins_z[i]) + double(aabb_maxs_z[i]));

    double mx = (cx - mesh_min_x) / mesh_width_x;
    double my = (cy - mesh_min_y) / mesh_width_y;
    double mz = (cz - mesh_min_z) / mesh_width_z;

    if (dim == 2) {
        morton_codes[i] = gpu_morton_2d(mx, my);
    } else {
        morton_codes[i] = gpu_morton_3d(mx, my, mz);
    }
    box_ids[i] = i;
}

__device__ int delta_func(
    const uint64_t* sorted_codes,
    int i, uint64_t code_i, int j, int n)
{
    if (j < 0 || j >= n)
        return -1;
    uint64_t code_j = sorted_codes[j];
    if (code_i == code_j) {
        return 32 + __clz(i ^ j);
    }
    return __clzll(code_i ^ code_j);
}

__global__ void build_hierarchy_kernel(
    const uint64_t* sorted_codes,
    const int32_t* sorted_ids,
    const float* aabb_mins_x, const float* aabb_mins_y, const float* aabb_mins_z,
    const float* aabb_maxs_x, const float* aabb_maxs_y, const float* aabb_maxs_z,
    GpuNode* nodes,
    int32_t* rightmost_leaves,
    int32_t* visitation_counts,
    int32_t* left_ranges,
    int32_t* right_ranges,
    int32_t* root_idx,
    const int N_LEAVES)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_LEAVES)
        return;

    const int LEAF_OFFSET = N_LEAVES - 1;

    {
        GpuNode leaf;
        leaf.aabb_min_x = aabb_mins_x[i];
        leaf.aabb_min_y = aabb_mins_y[i];
        leaf.aabb_min_z = aabb_mins_z[i];
        leaf.aabb_max_x = aabb_maxs_x[i];
        leaf.aabb_max_y = aabb_maxs_y[i];
        leaf.aabb_max_z = aabb_maxs_z[i];
        leaf.primitive_id = sorted_ids[i];
        leaf.is_inner_marker = 0;
        leaf.left = 0;
        leaf.right = 0;
        nodes[LEAF_OFFSET + i] = leaf;
        rightmost_leaves[LEAF_OFFSET + i] = i;
    }

    int left_key = i;
    int right_key = i;
    int current_node = LEAF_OFFSET + i;

    while (true) {
        const bool is_child_a = (left_key == 0)
            || (right_key != N_LEAVES - 1
                && delta_func(sorted_codes, right_key,
                       sorted_codes[right_key], right_key + 1, N_LEAVES)
                    > delta_func(sorted_codes, left_key - 1,
                        sorted_codes[left_key - 1], left_key, N_LEAVES));
        const int parent = is_child_a ? right_key : left_key - 1;

        if (is_child_a) {
            nodes[parent].left = current_node;
            left_ranges[parent] = left_key;
        } else {
            nodes[parent].right = current_node;
            right_ranges[parent] = right_key;
        }

        __threadfence();

        if (atomicAdd(&visitation_counts[parent], 1) == 0) {
            break;
        }

        __threadfence();

        const GpuNode& child_a = nodes[nodes[parent].left];
        const GpuNode& child_b = nodes[nodes[parent].right];

        nodes[parent].aabb_min_x = fminf(child_a.aabb_min_x, child_b.aabb_min_x);
        nodes[parent].aabb_min_y = fminf(child_a.aabb_min_y, child_b.aabb_min_y);
        nodes[parent].aabb_min_z = fminf(child_a.aabb_min_z, child_b.aabb_min_z);
        nodes[parent].aabb_max_x = fmaxf(child_a.aabb_max_x, child_b.aabb_max_x);
        nodes[parent].aabb_max_y = fmaxf(child_a.aabb_max_y, child_b.aabb_max_y);
        nodes[parent].aabb_max_z = fmaxf(child_a.aabb_max_z, child_b.aabb_max_z);
        nodes[parent].is_inner_marker = 1;

        rightmost_leaves[parent] = max(
            rightmost_leaves[nodes[parent].left],
            rightmost_leaves[nodes[parent].right]);

        left_key = left_ranges[parent];
        right_key = right_ranges[parent];
        current_node = parent;

        if (left_key == 0 && right_key == N_LEAVES - 1) {
            int expected = -1;
            atomicCAS(root_idx, expected, current_node);
            break;
        }
    }
}

template <bool triangular, bool count_only>
__global__ void traverse_lbvh_kernel(
    const GpuNode* __restrict__ source_bvh,
    const GpuNode* __restrict__ target_bvh,
    const int32_t* __restrict__ rightmost_leaves,
    const int num_source_leaves,
    const int source_leaf_offset,
    const int num_target_nodes,
    const int target_root,
    const int32_t* __restrict__ edge_v0,
    const int32_t* __restrict__ edge_v1,
    const int32_t* __restrict__ face_v0,
    const int32_t* __restrict__ face_v1,
    const int32_t* __restrict__ face_v2,
    const int filter_type,
    const int32_t* __restrict__ offsets,
    int32_t* candidate_counts,
    CandidatePair* candidate_pairs)
{
    const int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_source_leaves)
        return;

    const GpuNode& query = source_bvh[source_leaf_offset + query_idx];

    constexpr int MAX_STACK = 64;
    int stack[MAX_STACK];
    int stack_ptr = 0;
    stack[stack_ptr++] = -1;

    int node_idx = target_root;
    int local_count = 0;

    auto try_emit = [&](int32_t id_a, int32_t id_b) {
        if (filter_type == 1) {
            if (id_b == edge_v0[id_a] || id_b == edge_v1[id_a]) return;
        } else if (filter_type == 2) {
            int32_t ea0 = edge_v0[id_a], ea1 = edge_v1[id_a];
            int32_t eb0 = edge_v0[id_b], eb1 = edge_v1[id_b];
            if (ea0 == eb0 || ea0 == eb1 || ea1 == eb0 || ea1 == eb1) return;
        } else if (filter_type == 3) {
            if (id_a == face_v0[id_b] || id_a == face_v1[id_b] || id_a == face_v2[id_b]) return;
        } else if (filter_type == 4) {
            int32_t e0 = edge_v0[id_b], e1 = edge_v1[id_b];
            int32_t f0 = face_v0[id_a], f1 = face_v1[id_a], f2 = face_v2[id_a];
            if (e0 == f0 || e0 == f1 || e0 == f2 || e1 == f0 || e1 == f1 || e1 == f2) return;
        } else if (filter_type == 5) {
            int32_t fa0 = face_v0[id_a], fa1 = face_v1[id_a], fa2 = face_v2[id_a];
            int32_t fb0 = face_v0[id_b], fb1 = face_v1[id_b], fb2 = face_v2[id_b];
            if (fa0 == fb0 || fa0 == fb1 || fa0 == fb2 || fa1 == fb0 || fa1 == fb1 || fa1 == fb2
                || fa2 == fb0 || fa2 == fb1 || fa2 == fb2) return;
        }

        if constexpr (!count_only) {
            candidate_pairs[offsets[query_idx] + local_count].id_a = id_a;
            candidate_pairs[offsets[query_idx] + local_count].id_b = id_b;
        }
        local_count++;
    };

    do {
        if (num_target_nodes == 1) {
            if (!triangular && gpu_nodes_intersect(query, target_bvh[0])) {
                try_emit(query.primitive_id, target_bvh[0].primitive_id);
            }
            break;
        }

        const GpuNode& node = target_bvh[node_idx];
        const GpuNode& child_l = target_bvh[node.left];
        const GpuNode& child_r = target_bvh[node.right];

        bool intersects_l = gpu_nodes_intersect(query, child_l);
        bool intersects_r = gpu_nodes_intersect(query, child_r);

        if constexpr (triangular) {
            if (intersects_l && rightmost_leaves[node.left] <= query_idx)
                intersects_l = false;
            if (intersects_r && rightmost_leaves[node.right] <= query_idx)
                intersects_r = false;
        }

        if (intersects_l && gpu_node_is_leaf(child_l)) {
            try_emit(query.primitive_id, child_l.primitive_id);
        }
        if (intersects_r && gpu_node_is_leaf(child_r)) {
            try_emit(query.primitive_id, child_r.primitive_id);
        }

        bool traverse_l = intersects_l && gpu_node_is_inner(child_l);
        bool traverse_r = intersects_r && gpu_node_is_inner(child_r);

        if (!traverse_l && !traverse_r) {
            node_idx = stack[--stack_ptr];
        } else {
            node_idx = traverse_l ? node.left : node.right;
            if (traverse_l && traverse_r) {
                stack[stack_ptr++] = node.right;
            }
        }
    } while (node_idx != -1);

    candidate_counts[query_idx] = local_count;
}

} // anonymous namespace

struct GPU_LBVH::Impl {
    thrust::device_vector<GpuNode> vertex_nodes;
    thrust::device_vector<int32_t> vertex_rightmost;

    thrust::device_vector<GpuNode> edge_nodes;
    thrust::device_vector<int32_t> edge_rightmost;

    thrust::device_vector<GpuNode> face_nodes;
    thrust::device_vector<int32_t> face_rightmost;

    std::vector<int32_t> h_edge_v0;
    std::vector<int32_t> h_edge_v1;
    std::vector<int32_t> h_face_v0;
    std::vector<int32_t> h_face_v1;
    std::vector<int32_t> h_face_v2;

    thrust::device_vector<int32_t> d_edge_v0;
    thrust::device_vector<int32_t> d_edge_v1;
    thrust::device_vector<int32_t> d_face_v0;
    thrust::device_vector<int32_t> d_face_v1;
    thrust::device_vector<int32_t> d_face_v2;

    bool has_vertices = false;
    bool has_edges = false;
    bool has_faces = false;
    uint8_t dim = 0;

    int32_t vertex_root = 0;
    int32_t edge_root = 0;
    int32_t face_root = 0;

    void build_bvh(
        const AABBs& boxes,
        thrust::device_vector<GpuNode>& nodes,
        thrust::device_vector<int32_t>& rightmost,
        int32_t& root_out,
        const Eigen::Array3d& mesh_min,
        const Eigen::Array3d& mesh_max)
    {
        const int n = boxes.size();
        if (n == 0)
            return;

        const int num_nodes = 2 * n - 1;
        nodes.resize(num_nodes);
        rightmost.resize(num_nodes);

        thrust::device_vector<float> d_mins_x(n), d_mins_y(n), d_mins_z(n);
        thrust::device_vector<float> d_maxs_x(n), d_maxs_y(n), d_maxs_z(n);

        {
            thrust::host_vector<float> h_mx(n), h_my(n), h_mz(n);
            thrust::host_vector<float> h_Mx(n), h_My(n), h_Mz(n);
            for (int i = 0; i < n; i++) {
                h_mx[i] = std::nextafter(float(boxes[i].min.x()), -std::numeric_limits<float>::infinity());
                h_my[i] = std::nextafter(float(boxes[i].min.y()), -std::numeric_limits<float>::infinity());
                h_mz[i] = std::nextafter(float(boxes[i].min.z()), -std::numeric_limits<float>::infinity());
                h_Mx[i] = std::nextafter(float(boxes[i].max.x()), std::numeric_limits<float>::infinity());
                h_My[i] = std::nextafter(float(boxes[i].max.y()), std::numeric_limits<float>::infinity());
                h_Mz[i] = std::nextafter(float(boxes[i].max.z()), std::numeric_limits<float>::infinity());
            }
            d_mins_x = h_mx; d_mins_y = h_my; d_mins_z = h_mz;
            d_maxs_x = h_Mx; d_maxs_y = h_My; d_maxs_z = h_Mz;
        }

        thrust::device_vector<uint64_t> d_morton_codes(n);
        thrust::device_vector<int32_t> d_box_ids(n);

        const Eigen::Array3d mesh_width = mesh_max - mesh_min;
        double wx = mesh_width.x() > 0 ? mesh_width.x() : 1.0;
        double wy = mesh_width.y() > 0 ? mesh_width.y() : 1.0;
        double wz = mesh_width.z() > 0 ? mesh_width.z() : 1.0;

        int block = 256;
        int grid = (n + block - 1) / block;
        compute_morton_codes_kernel<<<grid, block>>>(
            n,
            mesh_min.x(), mesh_min.y(), mesh_min.z(),
            wx, wy, wz,
            thrust::raw_pointer_cast(d_mins_x.data()),
            thrust::raw_pointer_cast(d_mins_y.data()),
            thrust::raw_pointer_cast(d_mins_z.data()),
            thrust::raw_pointer_cast(d_maxs_x.data()),
            thrust::raw_pointer_cast(d_maxs_y.data()),
            thrust::raw_pointer_cast(d_maxs_z.data()),
            thrust::raw_pointer_cast(d_morton_codes.data()),
            thrust::raw_pointer_cast(d_box_ids.data()),
            dim);
        cudaDeviceSynchronize();

        thrust::sort_by_key(d_morton_codes.begin(), d_morton_codes.end(), d_box_ids.begin());

        thrust::device_vector<float> sorted_mins_x(n), sorted_mins_y(n), sorted_mins_z(n);
        thrust::device_vector<float> sorted_maxs_x(n), sorted_maxs_y(n), sorted_maxs_z(n);

        thrust::gather(d_box_ids.begin(), d_box_ids.end(), d_mins_x.begin(), sorted_mins_x.begin());
        thrust::gather(d_box_ids.begin(), d_box_ids.end(), d_mins_y.begin(), sorted_mins_y.begin());
        thrust::gather(d_box_ids.begin(), d_box_ids.end(), d_mins_z.begin(), sorted_mins_z.begin());
        thrust::gather(d_box_ids.begin(), d_box_ids.end(), d_maxs_x.begin(), sorted_maxs_x.begin());
        thrust::gather(d_box_ids.begin(), d_box_ids.end(), d_maxs_y.begin(), sorted_maxs_y.begin());
        thrust::gather(d_box_ids.begin(), d_box_ids.end(), d_maxs_z.begin(), sorted_maxs_z.begin());

        thrust::device_vector<int32_t> d_visitation_counts(num_nodes, 0);
        thrust::device_vector<int32_t> d_left_ranges(num_nodes);
        thrust::device_vector<int32_t> d_right_ranges(num_nodes);
        thrust::device_vector<int32_t> d_root_idx(1, -1);

        build_hierarchy_kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(d_morton_codes.data()),
            thrust::raw_pointer_cast(d_box_ids.data()),
            thrust::raw_pointer_cast(sorted_mins_x.data()),
            thrust::raw_pointer_cast(sorted_mins_y.data()),
            thrust::raw_pointer_cast(sorted_mins_z.data()),
            thrust::raw_pointer_cast(sorted_maxs_x.data()),
            thrust::raw_pointer_cast(sorted_maxs_y.data()),
            thrust::raw_pointer_cast(sorted_maxs_z.data()),
            thrust::raw_pointer_cast(nodes.data()),
            thrust::raw_pointer_cast(rightmost.data()),
            thrust::raw_pointer_cast(d_visitation_counts.data()),
            thrust::raw_pointer_cast(d_left_ranges.data()),
            thrust::raw_pointer_cast(d_right_ranges.data()),
            thrust::raw_pointer_cast(d_root_idx.data()),
            n);
        cudaDeviceSynchronize();

        int32_t h_root;
        thrust::copy(d_root_idx.begin(), d_root_idx.end(), &h_root);
        if (h_root < 0) h_root = 0;
        root_out = h_root;
    }

    template <bool triangular>
    void launch_traverse(
        const thrust::device_vector<GpuNode>& source,
        const thrust::device_vector<GpuNode>& target,
        const thrust::device_vector<int32_t>& rightmost,
        int target_root,
        int filter_type,
        int32_t* d_counts,
        const int32_t* d_offsets,
        CandidatePair* d_pairs) const
    {
        const int source_leaf_offset = source.size() / 2;
        const int num_source_leaves = source_leaf_offset + 1;
        const int num_target_nodes = target.size();
        const int block = 256;
        const int grid = (num_source_leaves + block - 1) / block;

        bool do_count = (d_pairs == nullptr);

        if (do_count) {
            traverse_lbvh_kernel<triangular, true><<<grid, block>>>(
                thrust::raw_pointer_cast(source.data()),
                thrust::raw_pointer_cast(target.data()),
                thrust::raw_pointer_cast(rightmost.data()),
                num_source_leaves, source_leaf_offset, num_target_nodes, target_root,
                d_edge_v0.empty() ? nullptr : thrust::raw_pointer_cast(d_edge_v0.data()),
                d_edge_v1.empty() ? nullptr : thrust::raw_pointer_cast(d_edge_v1.data()),
                d_face_v0.empty() ? nullptr : thrust::raw_pointer_cast(d_face_v0.data()),
                d_face_v1.empty() ? nullptr : thrust::raw_pointer_cast(d_face_v1.data()),
                d_face_v2.empty() ? nullptr : thrust::raw_pointer_cast(d_face_v2.data()),
                filter_type,
                d_offsets,
                d_counts,
                nullptr);
        } else {
            traverse_lbvh_kernel<triangular, false><<<grid, block>>>(
                thrust::raw_pointer_cast(source.data()),
                thrust::raw_pointer_cast(target.data()),
                thrust::raw_pointer_cast(rightmost.data()),
                num_source_leaves, source_leaf_offset, num_target_nodes, target_root,
                d_edge_v0.empty() ? nullptr : thrust::raw_pointer_cast(d_edge_v0.data()),
                d_edge_v1.empty() ? nullptr : thrust::raw_pointer_cast(d_edge_v1.data()),
                d_face_v0.empty() ? nullptr : thrust::raw_pointer_cast(d_face_v0.data()),
                d_face_v1.empty() ? nullptr : thrust::raw_pointer_cast(d_face_v1.data()),
                d_face_v2.empty() ? nullptr : thrust::raw_pointer_cast(d_face_v2.data()),
                filter_type,
                d_offsets,
                d_counts,
                d_pairs);
        }
        cudaDeviceSynchronize();
    }

    std::vector<CandidatePair> traverse(
        const thrust::device_vector<GpuNode>& source,
        const thrust::device_vector<GpuNode>& target,
        const thrust::device_vector<int32_t>& rightmost,
        int target_root,
        bool triangular,
        int filter_type) const
    {
        if (source.empty() || target.empty())
            return {};

        const int source_leaf_offset = source.size() / 2;
        const int num_source_leaves = source_leaf_offset + 1;

        thrust::device_vector<int32_t> d_counts(num_source_leaves, 0);
        thrust::device_vector<int32_t> d_offsets(num_source_leaves, 0);

        if (triangular) {
            launch_traverse<true>(source, target, rightmost, target_root, filter_type,
                thrust::raw_pointer_cast(d_counts.data()),
                thrust::raw_pointer_cast(d_offsets.data()),
                nullptr);
        } else {
            launch_traverse<false>(source, target, rightmost, target_root, filter_type,
                thrust::raw_pointer_cast(d_counts.data()),
                thrust::raw_pointer_cast(d_offsets.data()),
                nullptr);
        }

        thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin());

        thrust::host_vector<int32_t> h_counts = d_counts;
        int total = 0;
        for (int i = 0; i < num_source_leaves; i++) {
            total += h_counts[i];
        }

        if (total == 0)
            return {};

        thrust::device_vector<CandidatePair> d_pairs(total);

        if (triangular) {
            launch_traverse<true>(source, target, rightmost, target_root, filter_type,
                thrust::raw_pointer_cast(d_counts.data()),
                thrust::raw_pointer_cast(d_offsets.data()),
                thrust::raw_pointer_cast(d_pairs.data()));
        } else {
            launch_traverse<false>(source, target, rightmost, target_root, filter_type,
                thrust::raw_pointer_cast(d_counts.data()),
                thrust::raw_pointer_cast(d_offsets.data()),
                thrust::raw_pointer_cast(d_pairs.data()));
        }

        thrust::host_vector<CandidatePair> h_pairs = d_pairs;
        return std::vector<CandidatePair>(h_pairs.begin(), h_pairs.end());
    }
};

GPU_LBVH::GPU_LBVH() : BroadPhase(), impl(std::make_unique<Impl>()) {}

GPU_LBVH::~GPU_LBVH() = default;

void GPU_LBVH::build(
    Eigen::ConstRef<Eigen::MatrixXi> edges,
    Eigen::ConstRef<Eigen::MatrixXi> faces)
{
    IPC_TOOLKIT_PROFILE_BLOCK("GPU_LBVH::build");

    BroadPhase::build(edges, faces);

    if (vertex_boxes.empty())
        return;

    impl->dim = dim;
    impl->has_vertices = !vertex_boxes.empty();
    impl->has_edges = !edge_boxes.empty();
    impl->has_faces = !face_boxes.empty();

    Eigen::Array3d mesh_min, mesh_max;
    compute_mesh_aabb(mesh_min, mesh_max);

    if (!vertex_boxes.empty()) {
        impl->build_bvh(vertex_boxes, impl->vertex_nodes, impl->vertex_rightmost, impl->vertex_root, mesh_min, mesh_max);
    }
    if (!edge_boxes.empty()) {
        impl->build_bvh(edge_boxes, impl->edge_nodes, impl->edge_rightmost, impl->edge_root, mesh_min, mesh_max);
    }
    if (!face_boxes.empty()) {
        impl->build_bvh(face_boxes, impl->face_nodes, impl->face_rightmost, impl->face_root, mesh_min, mesh_max);
    }

    impl->h_edge_v0.resize(edges.rows());
    impl->h_edge_v1.resize(edges.rows());
    thrust::host_vector<int32_t> h_ev0(edges.rows()), h_ev1(edges.rows());
    for (int i = 0; i < edges.rows(); i++) {
        impl->h_edge_v0[i] = edges(i, 0);
        impl->h_edge_v1[i] = edges(i, 1);
        h_ev0[i] = edges(i, 0);
        h_ev1[i] = edges(i, 1);
    }
    impl->d_edge_v0 = h_ev0;
    impl->d_edge_v1 = h_ev1;

    if (faces.rows() > 0) {
        impl->h_face_v0.resize(faces.rows());
        impl->h_face_v1.resize(faces.rows());
        impl->h_face_v2.resize(faces.rows());
        thrust::host_vector<int32_t> h_fv0(faces.rows()), h_fv1(faces.rows()), h_fv2(faces.rows());
        for (int i = 0; i < faces.rows(); i++) {
            impl->h_face_v0[i] = faces(i, 0);
            impl->h_face_v1[i] = faces(i, 1);
            impl->h_face_v2[i] = faces(i, 2);
            h_fv0[i] = faces(i, 0);
            h_fv1[i] = faces(i, 1);
            h_fv2[i] = faces(i, 2);
        }
        impl->d_face_v0 = h_fv0;
        impl->d_face_v1 = h_fv1;
        impl->d_face_v2 = h_fv2;
    }

    vertex_boxes.clear();
    edge_boxes.clear();
    face_boxes.clear();
}

void GPU_LBVH::clear()
{
    BroadPhase::clear();
    impl = std::make_unique<Impl>();
}

void GPU_LBVH::detect_vertex_vertex_candidates(
    std::vector<VertexVertexCandidate>& candidates) const
{
    if (impl->vertex_nodes.size() <= 1)
        return;
    IPC_TOOLKIT_PROFILE_BLOCK("GPU_LBVH::detect_vertex_vertex_candidates");
    auto pairs = impl->traverse(impl->vertex_nodes, impl->vertex_nodes, impl->vertex_rightmost, impl->vertex_root, true, 0);
    for (auto& p : pairs) {
        if (can_vertices_collide(p.id_a, p.id_b))
            candidates.emplace_back(p.id_a, p.id_b);
    }
}

void GPU_LBVH::detect_edge_vertex_candidates(
    std::vector<EdgeVertexCandidate>& candidates) const
{
    if (!impl->has_edges || !impl->has_vertices)
        return;
    IPC_TOOLKIT_PROFILE_BLOCK("GPU_LBVH::detect_edge_vertex_candidates");
    auto pairs = impl->traverse(impl->edge_nodes, impl->vertex_nodes, impl->vertex_rightmost, impl->vertex_root, false, 1);
    for (auto& p : pairs) {
        if (can_edge_vertex_collide(p.id_a, p.id_b))
            candidates.emplace_back(p.id_a, p.id_b);
    }
}

void GPU_LBVH::detect_edge_edge_candidates(
    std::vector<EdgeEdgeCandidate>& candidates) const
{
    if (impl->edge_nodes.size() <= 1)
        return;
    IPC_TOOLKIT_PROFILE_BLOCK("GPU_LBVH::detect_edge_edge_candidates");
    auto pairs = impl->traverse(impl->edge_nodes, impl->edge_nodes, impl->edge_rightmost, impl->edge_root, true, 2);
    for (auto& p : pairs) {
        if (can_edges_collide(p.id_a, p.id_b))
            candidates.emplace_back(p.id_a, p.id_b);
    }
}

void GPU_LBVH::detect_face_vertex_candidates(
    std::vector<FaceVertexCandidate>& candidates) const
{
    if (!impl->has_faces || !impl->has_vertices)
        return;
    IPC_TOOLKIT_PROFILE_BLOCK("GPU_LBVH::detect_face_vertex_candidates");
    auto pairs = impl->traverse(impl->vertex_nodes, impl->face_nodes, impl->face_rightmost, impl->face_root, false, 3);
    for (auto& p : pairs) {
        if (can_face_vertex_collide(p.id_b, p.id_a))
            candidates.emplace_back(p.id_b, p.id_a);
    }
}

void GPU_LBVH::detect_edge_face_candidates(
    std::vector<EdgeFaceCandidate>& candidates) const
{
    if (!impl->has_edges || !impl->has_faces)
        return;
    IPC_TOOLKIT_PROFILE_BLOCK("GPU_LBVH::detect_edge_face_candidates");
    auto pairs = impl->traverse(impl->face_nodes, impl->edge_nodes, impl->edge_rightmost, impl->edge_root, false, 4);
    for (auto& p : pairs) {
        if (can_edge_face_collide(p.id_b, p.id_a))
            candidates.emplace_back(p.id_b, p.id_a);
    }
}

void GPU_LBVH::detect_face_face_candidates(
    std::vector<FaceFaceCandidate>& candidates) const
{
    if (impl->face_nodes.size() <= 1)
        return;
    IPC_TOOLKIT_PROFILE_BLOCK("GPU_LBVH::detect_face_face_candidates");
    auto pairs = impl->traverse(impl->face_nodes, impl->face_nodes, impl->face_rightmost, impl->face_root, true, 5);
    for (auto& p : pairs) {
        if (can_faces_collide(p.id_a, p.id_b))
            candidates.emplace_back(p.id_a, p.id_b);
    }
}

bool GPU_LBVH::can_edge_vertex_collide(size_t ei, size_t vi) const
{
    if (ei >= impl->h_edge_v0.size()) return true;
    int32_t e0 = impl->h_edge_v0[ei], e1 = impl->h_edge_v1[ei];
    return vi != size_t(e0) && vi != size_t(e1)
        && (can_vertices_collide(vi, e0) || can_vertices_collide(vi, e1));
}

bool GPU_LBVH::can_edges_collide(size_t eai, size_t ebi) const
{
    if (eai >= impl->h_edge_v0.size()) return true;
    int32_t ea0 = impl->h_edge_v0[eai], ea1 = impl->h_edge_v1[eai];
    int32_t eb0 = impl->h_edge_v0[ebi], eb1 = impl->h_edge_v1[ebi];
    bool share = ea0 == eb0 || ea0 == eb1 || ea1 == eb0 || ea1 == eb1;
    return !share && (can_vertices_collide(ea0, eb0) || can_vertices_collide(ea0, eb1)
        || can_vertices_collide(ea1, eb0) || can_vertices_collide(ea1, eb1));
}

bool GPU_LBVH::can_face_vertex_collide(size_t fi, size_t vi) const
{
    if (fi >= impl->h_face_v0.size()) return true;
    int32_t f0 = impl->h_face_v0[fi], f1 = impl->h_face_v1[fi], f2 = impl->h_face_v2[fi];
    return vi != size_t(f0) && vi != size_t(f1) && vi != size_t(f2)
        && (can_vertices_collide(vi, f0) || can_vertices_collide(vi, f1) || can_vertices_collide(vi, f2));
}

bool GPU_LBVH::can_edge_face_collide(size_t ei, size_t fi) const
{
    if (ei >= impl->h_edge_v0.size() || fi >= impl->h_face_v0.size()) return true;
    int32_t e0 = impl->h_edge_v0[ei], e1 = impl->h_edge_v1[ei];
    int32_t f0 = impl->h_face_v0[fi], f1 = impl->h_face_v1[fi], f2 = impl->h_face_v2[fi];
    bool share = e0 == f0 || e0 == f1 || e0 == f2 || e1 == f0 || e1 == f1 || e1 == f2;
    return !share && (can_vertices_collide(e0, f0) || can_vertices_collide(e0, f1) || can_vertices_collide(e0, f2)
        || can_vertices_collide(e1, f0) || can_vertices_collide(e1, f1) || can_vertices_collide(e1, f2));
}

bool GPU_LBVH::can_faces_collide(size_t fai, size_t fbi) const
{
    if (fai >= impl->h_face_v0.size() || fbi >= impl->h_face_v0.size()) return true;
    int32_t fa0 = impl->h_face_v0[fai], fa1 = impl->h_face_v1[fai], fa2 = impl->h_face_v2[fai];
    int32_t fb0 = impl->h_face_v0[fbi], fb1 = impl->h_face_v1[fbi], fb2 = impl->h_face_v2[fbi];
    bool share = fa0 == fb0 || fa0 == fb1 || fa0 == fb2 || fa1 == fb0 || fa1 == fb1 || fa1 == fb2
        || fa2 == fb0 || fa2 == fb1 || fa2 == fb2;
    return !share && (can_vertices_collide(fa0, fb0) || can_vertices_collide(fa0, fb1) || can_vertices_collide(fa0, fb2)
        || can_vertices_collide(fa1, fb0) || can_vertices_collide(fa1, fb1) || can_vertices_collide(fa1, fb2)
        || can_vertices_collide(fa2, fb0) || can_vertices_collide(fa2, fb1) || can_vertices_collide(fa2, fb2));
}

} // namespace ipc

#endif
