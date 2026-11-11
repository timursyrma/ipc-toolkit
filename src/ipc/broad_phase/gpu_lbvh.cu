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
#include <thrust/reduce.h>
#include <thrust/fill.h>

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

// Squared distance between segments [p1,q1] and [p2,q2] (Ericson,
// "Real-Time Collision Detection", 5.1.9). Used by the exact EE culling:
// capsule-vs-capsule test at the leaf level is exactly this distance
// against dhat.
//
// Computed in double: the closest-point parameters come out of
// denom = a*e - b*b, which catastrophically cancels in float for short
// near-parallel edges (small dhat regime), overestimating the distance and
// culling live pairs. Doubles only run for leaf pairs that already passed
// the AABB test, so the FP64 throughput penalty is negligible against the
// memory-bound traversal.
// Float version: cheap, but its closest-point parameters can be off for
// short near-parallel edges, so its value is only trusted far from the
// threshold (see ee_culled()).
__device__ inline float seg_seg_dist_sq_f(
    const float3 p1, const float3 q1, const float3 p2, const float3 q2)
{
    const float3 d1 = { q1.x - p1.x, q1.y - p1.y, q1.z - p1.z };
    const float3 d2 = { q2.x - p2.x, q2.y - p2.y, q2.z - p2.z };
    const float3 r = { p1.x - p2.x, p1.y - p2.y, p1.z - p2.z };
    const float a = d1.x * d1.x + d1.y * d1.y + d1.z * d1.z;
    const float e = d2.x * d2.x + d2.y * d2.y + d2.z * d2.z;
    const float f = d2.x * r.x + d2.y * r.y + d2.z * r.z;

    float s, t;
    constexpr float EPS = 1e-20f;
    if (a <= EPS && e <= EPS) {
        s = t = 0.0f;
    } else if (a <= EPS) {
        s = 0.0f;
        t = fminf(fmaxf(f / e, 0.0f), 1.0f);
    } else {
        const float c = d1.x * r.x + d1.y * r.y + d1.z * r.z;
        if (e <= EPS) {
            t = 0.0f;
            s = fminf(fmaxf(-c / a, 0.0f), 1.0f);
        } else {
            const float b = d1.x * d2.x + d1.y * d2.y + d1.z * d2.z;
            const float denom = a * e - b * b;
            s = denom > EPS
                ? fminf(fmaxf((b * f - c * e) / denom, 0.0f), 1.0f)
                : 0.0f;
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = fminf(fmaxf(-c / a, 0.0f), 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = fminf(fmaxf((b - c) / a, 0.0f), 1.0f);
            }
        }
    }

    const float3 c1 = { p1.x + d1.x * s, p1.y + d1.y * s, p1.z + d1.z * s };
    const float3 c2 = { p2.x + d2.x * t, p2.y + d2.y * t, p2.z + d2.z * t };
    const float3 v = { c1.x - c2.x, c1.y - c2.y, c1.z - c2.z };
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ inline double seg_seg_dist_sq(
    const float3 p1f, const float3 q1f, const float3 p2f, const float3 q2f)
{
    const double3 p1 = { p1f.x, p1f.y, p1f.z };
    const double3 q1 = { q1f.x, q1f.y, q1f.z };
    const double3 p2 = { p2f.x, p2f.y, p2f.z };
    const double3 q2 = { q2f.x, q2f.y, q2f.z };

    const double3 d1 = { q1.x - p1.x, q1.y - p1.y, q1.z - p1.z };
    const double3 d2 = { q2.x - p2.x, q2.y - p2.y, q2.z - p2.z };
    const double3 r = { p1.x - p2.x, p1.y - p2.y, p1.z - p2.z };
    const double a = d1.x * d1.x + d1.y * d1.y + d1.z * d1.z;
    const double e = d2.x * d2.x + d2.y * d2.y + d2.z * d2.z;
    const double f = d2.x * r.x + d2.y * r.y + d2.z * r.z;

    double s, t;
    constexpr double EPS = 1e-30;
    if (a <= EPS && e <= EPS) {
        s = t = 0.0;
    } else if (a <= EPS) {
        s = 0.0;
        t = fmin(fmax(f / e, 0.0), 1.0);
    } else {
        const double c = d1.x * r.x + d1.y * r.y + d1.z * r.z;
        if (e <= EPS) {
            t = 0.0;
            s = fmin(fmax(-c / a, 0.0), 1.0);
        } else {
            const double b = d1.x * d2.x + d1.y * d2.y + d1.z * d2.z;
            const double denom = a * e - b * b;
            s = denom > EPS ? fmin(fmax((b * f - c * e) / denom, 0.0), 1.0)
                            : 0.0;
            t = (b * s + f) / e;
            if (t < 0.0) {
                t = 0.0;
                s = fmin(fmax(-c / a, 0.0), 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = fmin(fmax((b - c) / a, 0.0), 1.0);
            }
        }
    }

    const double3 c1 = { p1.x + d1.x * s, p1.y + d1.y * s, p1.z + d1.z * s };
    const double3 c2 = { p2.x + d2.x * t, p2.y + d2.y * t, p2.z + d2.z * t };
    const double3 v = { c1.x - c2.x, c1.y - c2.y, c1.z - c2.z };
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// Two-stage EE cull: the float distance decides only when it is at least
// 2x the threshold (4x in squared terms) -- far outside its observed error
// -- and the uncertainty band falls through to the double computation.
// TODO(research): replace the 2x guard band with a certified forward error
// bound on the float closest-point computation.
__device__ inline bool ee_culled(
    const float3 p1, const float3 q1, const float3 p2, const float3 q2,
    const float cull_sq)
{
    const float d_f = seg_seg_dist_sq_f(p1, q1, p2, q2);
    if (d_f <= cull_sq) {
        return false; // emitting extra is always safe
    }
    if (d_f > 4.0f * cull_sq) {
        return true; // far beyond the float error band
    }
    return seg_seg_dist_sq(p1, q1, p2, q2) > double(cull_sq);
}

// Point-point: a plain difference of squares has no parameter search, so
// the float error is tiny and the host-side threshold margin covers it.
__device__ inline bool vv_culled(
    const float3 p, const float3 q, const float cull_sq)
{
    const float3 v = { p.x - q.x, p.y - q.y, p.z - q.z };
    return v.x * v.x + v.y * v.y + v.z * v.z > cull_sq;
}

// Point-segment squared distance (scalar-templated: float fast path,
// double for the uncertainty band).
template <typename T>
__device__ inline T pt_seg_dist_sq_t(
    const float3 pf, const float3 af, const float3 bf)
{
    const T px = pf.x, py = pf.y, pz = pf.z;
    const T ax = af.x, ay = af.y, az = af.z;
    const T dx = T(bf.x) - ax, dy = T(bf.y) - ay, dz = T(bf.z) - az;
    const T len_sq = dx * dx + dy * dy + dz * dz;
    T t = len_sq > T(0)
        ? ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / len_sq
        : T(0);
    t = min(max(t, T(0)), T(1));
    const T vx = px - (ax + t * dx), vy = py - (ay + t * dy),
            vz = pz - (az + t * dz);
    return vx * vx + vy * vy + vz * vz;
}

__device__ inline bool ev_culled(
    const float3 p, const float3 a, const float3 b, const float cull_sq)
{
    const float d_f = pt_seg_dist_sq_t<float>(p, a, b);
    if (d_f <= cull_sq) {
        return false;
    }
    if (d_f > 4.0f * cull_sq) {
        return true;
    }
    return pt_seg_dist_sq_t<double>(p, a, b) > double(cull_sq);
}

// Point-triangle squared distance (Ericson, "Real-Time Collision
// Detection", 5.1.5), scalar-templated like pt_seg_dist_sq_t.
template <typename T>
__device__ inline T pt_tri_dist_sq_t(
    const float3 pf, const float3 af, const float3 bf, const float3 cf)
{
    const T px = pf.x, py = pf.y, pz = pf.z;
    const T ax = af.x, ay = af.y, az = af.z;
    const T abx = T(bf.x) - ax, aby = T(bf.y) - ay, abz = T(bf.z) - az;
    const T acx = T(cf.x) - ax, acy = T(cf.y) - ay, acz = T(cf.z) - az;
    const T apx = px - ax, apy = py - ay, apz = pz - az;

    const T d1 = abx * apx + aby * apy + abz * apz;
    const T d2 = acx * apx + acy * apy + acz * apz;

    T qx, qy, qz; // closest point
    if (d1 <= T(0) && d2 <= T(0)) {
        qx = ax, qy = ay, qz = az; // region A
    } else {
        const T bpx = px - T(bf.x), bpy = py - T(bf.y), bpz = pz - T(bf.z);
        const T d3 = abx * bpx + aby * bpy + abz * bpz;
        const T d4 = acx * bpx + acy * bpy + acz * bpz;
        if (d3 >= T(0) && d4 <= d3) {
            qx = bf.x, qy = bf.y, qz = bf.z; // region B
        } else {
            const T cpx = px - T(cf.x), cpy = py - T(cf.y),
                    cpz = pz - T(cf.z);
            const T d5 = abx * cpx + aby * cpy + abz * cpz;
            const T d6 = acx * cpx + acy * cpy + acz * cpz;
            const T vc = d1 * d4 - d3 * d2;
            const T vb = d5 * d2 - d1 * d6;
            const T va = d3 * d6 - d5 * d4;
            if (d6 >= T(0) && d5 <= d6) {
                qx = cf.x, qy = cf.y, qz = cf.z; // region C
            } else if (vc <= T(0) && d1 >= T(0) && d3 <= T(0)) {
                const T v = d1 / (d1 - d3); // edge AB
                qx = ax + v * abx, qy = ay + v * aby, qz = az + v * abz;
            } else if (vb <= T(0) && d2 >= T(0) && d6 <= T(0)) {
                const T w = d2 / (d2 - d6); // edge AC
                qx = ax + w * acx, qy = ay + w * acy, qz = az + w * acz;
            } else if (va <= T(0) && (d4 - d3) >= T(0) && (d5 - d6) >= T(0)) {
                const T w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); // edge BC
                qx = T(bf.x) + w * (T(cf.x) - T(bf.x));
                qy = T(bf.y) + w * (T(cf.y) - T(bf.y));
                qz = T(bf.z) + w * (T(cf.z) - T(bf.z));
            } else {
                const T denom = T(1) / (va + vb + vc); // interior
                const T v = vb * denom, w = vc * denom;
                qx = ax + abx * v + acx * w;
                qy = ay + aby * v + acy * w;
                qz = az + abz * v + acz * w;
            }
        }
    }

    const T vx = px - qx, vy = py - qy, vz = pz - qz;
    return vx * vx + vy * vy + vz * vz;
}

__device__ inline bool fv_culled(
    const float3 p, const float3 a, const float3 b, const float3 c,
    const float cull_sq)
{
    const float d_f = pt_tri_dist_sq_t<float>(p, a, b, c);
    if (d_f <= cull_sq) {
        return false;
    }
    if (d_f > 4.0f * cull_sq) {
        return true;
    }
    return pt_tri_dist_sq_t<double>(p, a, b, c) > double(cull_sq);
}

// Distance-aware node descent. The node boxes are inflated by dhat/2 on
// each side, so the uninflated per-axis gap is the inflated separation
// plus dhat; the Euclidean combination of the positive gaps lower-bounds
// the primitive distance, and a lower bound above dhat prunes the whole
// subtree. Strictly stronger than the plain overlap test (which only
// prunes when a single axis separates): boxes diagonal from each other
// get culled here even though every axis gap individually is below dhat.
__device__ inline bool gpu_nodes_within(
    const GpuNode& a, const GpuNode& b, const float dhat)
{
    float d2 = 0.0f, s;
    s = fmaxf(a.aabb_min_x - b.aabb_max_x, b.aabb_min_x - a.aabb_max_x)
        + dhat;
    if (s > 0.0f) {
        d2 += s * s;
    }
    s = fmaxf(a.aabb_min_y - b.aabb_max_y, b.aabb_min_y - a.aabb_max_y)
        + dhat;
    if (s > 0.0f) {
        d2 += s * s;
    }
    s = fmaxf(a.aabb_min_z - b.aabb_max_z, b.aabb_min_z - a.aabb_max_z)
        + dhat;
    if (s > 0.0f) {
        d2 += s * s;
    }
    return d2 <= dhat * dhat;
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
    const float3* __restrict__ verts,
    const float ee_cull_sq,
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
            // Exact dhat-culling: capsule-vs-capsule == segment distance.
            if (ee_cull_sq >= 0.0f
                && ee_culled(
                       verts[ea0], verts[ea1], verts[eb0], verts[eb1],
                       ee_cull_sq)) {
                return;
            }
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

template <bool triangular>
__global__ void traverse_single_pass_kernel(
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
    const float3* __restrict__ verts,
    const float ee_cull_sq,
    CandidatePair* __restrict__ candidate_pairs,
    int* __restrict__ global_counter,
    const int max_pairs)
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

    // Distance-aware descent threshold: the squared cull threshold carries
    // the host-side margin, so dhat here inherits it.
    const float cull_dhat = ee_cull_sq >= 0.0f ? sqrtf(ee_cull_sq) : -1.0f;

    auto try_emit = [&](int32_t id_a, int32_t id_b) {
        if (filter_type == 0) {
            if (ee_cull_sq >= 0.0f
                && vv_culled(verts[id_a], verts[id_b], ee_cull_sq)) {
                return;
            }
        } else if (filter_type == 1) {
            if (id_b == edge_v0[id_a] || id_b == edge_v1[id_a]) return;
            if (ee_cull_sq >= 0.0f
                && ev_culled(
                       verts[id_b], verts[edge_v0[id_a]],
                       verts[edge_v1[id_a]], ee_cull_sq)) {
                return;
            }
        } else if (filter_type == 2) {
            int32_t ea0 = edge_v0[id_a], ea1 = edge_v1[id_a];
            int32_t eb0 = edge_v0[id_b], eb1 = edge_v1[id_b];
            if (ea0 == eb0 || ea0 == eb1 || ea1 == eb0 || ea1 == eb1) return;
            // Exact dhat-culling: capsule-vs-capsule == segment distance.
            if (ee_cull_sq >= 0.0f
                && ee_culled(
                       verts[ea0], verts[ea1], verts[eb0], verts[eb1],
                       ee_cull_sq)) {
                return;
            }
        } else if (filter_type == 3) {
            if (id_a == face_v0[id_b] || id_a == face_v1[id_b] || id_a == face_v2[id_b]) return;
            if (ee_cull_sq >= 0.0f
                && fv_culled(
                       verts[id_a], verts[face_v0[id_b]],
                       verts[face_v1[id_b]], verts[face_v2[id_b]],
                       ee_cull_sq)) {
                return;
            }
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

        int idx = atomicAdd(global_counter, 1);
        if (idx < max_pairs) {
            candidate_pairs[idx].id_a = id_a;
            candidate_pairs[idx].id_b = id_b;
        }
    };

    auto node_hit = [&](const GpuNode& a, const GpuNode& b) {
        return cull_dhat >= 0.0f ? gpu_nodes_within(a, b, cull_dhat)
                                 : gpu_nodes_intersect(a, b);
    };

    do {
        if (num_target_nodes == 1) {
            if (!triangular && node_hit(query, target_bvh[0])) {
                try_emit(query.primitive_id, target_bvh[0].primitive_id);
            }
            break;
        }

        const GpuNode& node = target_bvh[node_idx];
        const GpuNode& child_l = target_bvh[node.left];
        const GpuNode& child_r = target_bvh[node.right];

        bool intersects_l = node_hit(query, child_l);
        bool intersects_r = node_hit(query, child_r);

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

    // Exact EE dhat-culling (see GPU_LBVH::exact_ee_culling). Set by the
    // protected build from the state staged by the public static build();
    // any other build path leaves it disabled.
    thrust::device_vector<float3> d_verts;
    float ee_cull_sq = -1.0f; ///< Active threshold; < 0 disables.

    int32_t vertex_root = 0;
    int32_t edge_root = 0;
    int32_t face_root = 0;

    mutable thrust::device_vector<int32_t> d_counts;
    mutable thrust::device_vector<int32_t> d_offsets;
    mutable thrust::device_vector<CandidatePair> d_pairs;
    mutable thrust::device_vector<int> d_single_counter;

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
                d_verts.empty() ? nullptr : thrust::raw_pointer_cast(d_verts.data()),
                (filter_type <= 3 && !d_verts.empty()) ? ee_cull_sq : -1.0f,
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
                d_verts.empty() ? nullptr : thrust::raw_pointer_cast(d_verts.data()),
                (filter_type <= 3 && !d_verts.empty()) ? ee_cull_sq : -1.0f,
                d_offsets,
                d_counts,
                d_pairs);
        }
        cudaDeviceSynchronize();
    }

    template <bool triangular>
    void launch_single_pass(
        const thrust::device_vector<GpuNode>& source,
        const thrust::device_vector<GpuNode>& target,
        const thrust::device_vector<int32_t>& rightmost,
        int target_root,
        int filter_type,
        CandidatePair* d_pairs,
        int* d_counter,
        int max_pairs) const
    {
        const int source_leaf_offset = source.size() / 2;
        const int num_source_leaves = source_leaf_offset + 1;
        const int num_target_nodes = target.size();
        const int block = 256;
        const int grid = (num_source_leaves + block - 1) / block;

        traverse_single_pass_kernel<triangular><<<grid, block>>>(
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
            d_verts.empty() ? nullptr : thrust::raw_pointer_cast(d_verts.data()),
            (filter_type <= 3 && !d_verts.empty()) ? ee_cull_sq : -1.0f,
            d_pairs,
            d_counter,
            max_pairs);
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

        d_counts.resize(num_source_leaves);
        d_offsets.resize(num_source_leaves);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

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

        int total = thrust::reduce(d_counts.begin(), d_counts.end(), 0, thrust::plus<int>());

        if (total == 0)
            return {};

        d_pairs.resize(total);

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

    /// Single-pass traversal: one kernel launch writing through a global
    /// atomic counter, instead of the count + write passes of traverse().
    /// The buffer starts at its high-water mark (or 8 pairs per leaf) and
    /// the kernel keeps counting past capacity, so an overflow re-runs once
    /// with the exact size; steady-state is a single pass. Used for the
    /// exact-culled EE path, where repeating the distance test in a second
    /// pass would double the most expensive part of the traversal.
    std::vector<CandidatePair> traverse_single(
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

        size_t capacity = std::max(
            { d_pairs.capacity(), size_t(1024),
              size_t(8) * size_t(num_source_leaves) });
        d_single_counter.resize(1);

        int count = 0;
        for (int pass = 0; pass < 2; ++pass) {
            if (d_pairs.size() < capacity)
                d_pairs.resize(capacity);
            thrust::fill(
                d_single_counter.begin(), d_single_counter.end(), 0);

            if (triangular) {
                launch_single_pass<true>(
                    source, target, rightmost, target_root, filter_type,
                    thrust::raw_pointer_cast(d_pairs.data()),
                    thrust::raw_pointer_cast(d_single_counter.data()),
                    int(capacity));
            } else {
                launch_single_pass<false>(
                    source, target, rightmost, target_root, filter_type,
                    thrust::raw_pointer_cast(d_pairs.data()),
                    thrust::raw_pointer_cast(d_single_counter.data()),
                    int(capacity));
            }

            thrust::copy(
                d_single_counter.begin(), d_single_counter.end(), &count);
            if (size_t(count) <= capacity)
                break;
            capacity = size_t(count); // exact size known; second pass fits
        }

        if (count == 0)
            return {};

        std::vector<CandidatePair> out(count);
        thrust::copy(d_pairs.begin(), d_pairs.begin() + count, out.begin());
        return out;
    }
};

GPU_LBVH::GPU_LBVH() : BroadPhase(), impl(std::make_unique<Impl>()) {}

GPU_LBVH::~GPU_LBVH() = default;

void GPU_LBVH::build(
    Eigen::ConstRef<Eigen::MatrixXd> vertices,
    Eigen::ConstRef<Eigen::MatrixXi> edges,
    Eigen::ConstRef<Eigen::MatrixXi> faces,
    const double inflation_radius)
{
    const bool cullable =
        exact_ee_culling && inflation_radius > 0 && vertices.rows() > 0;
    const double mu = warm_start_margin;
    const double dhat = 2 * inflation_radius;

    // Certified warm start: if nothing moved more than mu * dhat / 2 since
    // the cached build, the cached sets -- built with a (1 + mu) threshold
    // -- are supersets of the live sets, and the whole build is skipped.
    if (cullable && mu > 0 && m_cached_inflation == inflation_radius
        && m_cached_vertices.rows() == vertices.rows()
        && m_cached_num_edges == edges.rows()
        && m_cached_num_faces == faces.rows()) {
        const double max_disp = (vertices - m_cached_vertices)
                                    .rowwise()
                                    .norm()
                                    .maxCoeff();
        if (2 * max_disp <= mu * dhat) {
            m_reuse = true;
            return;
        }
    }
    m_reuse = false;

    if (cullable) {
        // Production convention: inflation_radius = dhat / 2
        // (NormalCollisions::build), so the exact threshold is
        // dhat = 2 * inflation_radius. The margin is the float-conservatism
        // slack (1e-2) plus the warm-start motion budget (mu). Staged in
        // class members: BroadPhase::build() calls clear(), which recreates
        // the Impl, before the protected build consumes this.
        const double margin = 1e-2 + mu;
        const double thr = dhat * (1 + margin);
        m_staged_ee_cull_sq = thr * thr;

        const bool is_3d = vertices.cols() >= 3;
        m_staged_verts.resize(3 * vertices.rows());
        for (Eigen::Index i = 0; i < vertices.rows(); ++i) {
            m_staged_verts[3 * i + 0] = static_cast<float>(vertices(i, 0));
            m_staged_verts[3 * i + 1] = static_cast<float>(vertices(i, 1));
            m_staged_verts[3 * i + 2] =
                is_3d ? static_cast<float>(vertices(i, 2)) : 0.0f;
        }

        // The tree inflation must also cover (1 + mu) * dhat, or the AABB
        // stage could prune a pair the certification relies on.
        BroadPhase::build(
            vertices, edges, faces, inflation_radius * (1 + margin));

        if (mu > 0) { // after the base build: its clear() resets the cache
            m_cached_vertices = vertices;
            m_cached_num_edges = edges.rows();
            m_cached_num_faces = faces.rows();
            m_cached_inflation = inflation_radius;
        }
        return;
    }

    BroadPhase::build(vertices, edges, faces, inflation_radius);
}

void GPU_LBVH::build(
    Eigen::ConstRef<Eigen::MatrixXi> edges,
    Eigen::ConstRef<Eigen::MatrixXi> faces)
{
    IPC_TOOLKIT_PROFILE_BLOCK("GPU_LBVH::build");

    // Consume the threshold staged by the static build() above; any other
    // path (dynamic t0/t1 build, build from boxes) leaves culling off.
    if (m_staged_ee_cull_sq > 0 && !m_staged_verts.empty()) {
        impl->ee_cull_sq = static_cast<float>(m_staged_ee_cull_sq);
        thrust::host_vector<float3> h_verts(m_staged_verts.size() / 3);
        for (size_t i = 0; i < h_verts.size(); ++i) {
            h_verts[i] = make_float3(
                m_staged_verts[3 * i + 0], m_staged_verts[3 * i + 1],
                m_staged_verts[3 * i + 2]);
        }
        impl->d_verts = h_verts;
    } else {
        impl->ee_cull_sq = -1.0f;
        impl->d_verts.clear();
    }
    m_staged_ee_cull_sq = -1;
    m_staged_verts.clear();

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
    m_reuse = false;
    m_has_vv = m_has_ev = m_has_ee = m_has_fv = false;
    m_cached_inflation = -1;
}

void GPU_LBVH::detect_vertex_vertex_candidates(
    std::vector<VertexVertexCandidate>& candidates) const
{
    if (m_reuse && m_has_vv) {
        candidates.insert(
            candidates.end(), m_cached_vv.begin(), m_cached_vv.end());
        return;
    }
    if (impl->vertex_nodes.size() <= 1)
        return;
    IPC_TOOLKIT_PROFILE_BLOCK("GPU_LBVH::detect_vertex_vertex_candidates");
    auto pairs = impl->ee_cull_sq >= 0.0f
        ? impl->traverse_single(impl->vertex_nodes, impl->vertex_nodes, impl->vertex_rightmost, impl->vertex_root, true, 0)
        : impl->traverse(impl->vertex_nodes, impl->vertex_nodes, impl->vertex_rightmost, impl->vertex_root, true, 0);
    for (auto& p : pairs) {
        if (can_vertices_collide(p.id_a, p.id_b))
            candidates.emplace_back(p.id_a, p.id_b);
    }
    if (warm_start_margin > 0 && impl->ee_cull_sq >= 0.0f) {
        m_cached_vv = candidates;
        m_has_vv = true;
    }
}

void GPU_LBVH::detect_edge_vertex_candidates(
    std::vector<EdgeVertexCandidate>& candidates) const
{
    if (m_reuse && m_has_ev) {
        candidates.insert(
            candidates.end(), m_cached_ev.begin(), m_cached_ev.end());
        return;
    }
    if (!impl->has_edges || !impl->has_vertices)
        return;
    IPC_TOOLKIT_PROFILE_BLOCK("GPU_LBVH::detect_edge_vertex_candidates");
    auto pairs = impl->ee_cull_sq >= 0.0f
        ? impl->traverse_single(impl->edge_nodes, impl->vertex_nodes, impl->vertex_rightmost, impl->vertex_root, false, 1)
        : impl->traverse(impl->edge_nodes, impl->vertex_nodes, impl->vertex_rightmost, impl->vertex_root, false, 1);
    for (auto& p : pairs) {
        if (can_edge_vertex_collide(p.id_a, p.id_b))
            candidates.emplace_back(p.id_a, p.id_b);
    }
    if (warm_start_margin > 0 && impl->ee_cull_sq >= 0.0f) {
        m_cached_ev = candidates;
        m_has_ev = true;
    }
}

void GPU_LBVH::detect_edge_edge_candidates(
    std::vector<EdgeEdgeCandidate>& candidates) const
{
    if (m_reuse && m_has_ee) {
        candidates.insert(
            candidates.end(), m_cached_ee.begin(), m_cached_ee.end());
        return;
    }
    if (impl->edge_nodes.size() <= 1)
        return;
    IPC_TOOLKIT_PROFILE_BLOCK("GPU_LBVH::detect_edge_edge_candidates");
    auto pairs = impl->ee_cull_sq >= 0.0f
        ? impl->traverse_single(impl->edge_nodes, impl->edge_nodes, impl->edge_rightmost, impl->edge_root, true, 2)
        : impl->traverse(impl->edge_nodes, impl->edge_nodes, impl->edge_rightmost, impl->edge_root, true, 2);
    for (auto& p : pairs) {
        if (can_edges_collide(p.id_a, p.id_b))
            candidates.emplace_back(p.id_a, p.id_b);
    }
    if (warm_start_margin > 0 && impl->ee_cull_sq >= 0.0f) {
        m_cached_ee = candidates;
        m_has_ee = true;
    }
}

void GPU_LBVH::detect_face_vertex_candidates(
    std::vector<FaceVertexCandidate>& candidates) const
{
    if (m_reuse && m_has_fv) {
        candidates.insert(
            candidates.end(), m_cached_fv.begin(), m_cached_fv.end());
        return;
    }
    if (!impl->has_faces || !impl->has_vertices)
        return;
    IPC_TOOLKIT_PROFILE_BLOCK("GPU_LBVH::detect_face_vertex_candidates");
    auto pairs = impl->ee_cull_sq >= 0.0f
        ? impl->traverse_single(impl->vertex_nodes, impl->face_nodes, impl->face_rightmost, impl->face_root, false, 3)
        : impl->traverse(impl->vertex_nodes, impl->face_nodes, impl->face_rightmost, impl->face_root, false, 3);
    for (auto& p : pairs) {
        if (can_face_vertex_collide(p.id_b, p.id_a))
            candidates.emplace_back(p.id_b, p.id_a);
    }
    if (warm_start_margin > 0 && impl->ee_cull_sq >= 0.0f) {
        m_cached_fv = candidates;
        m_has_fv = true;
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
