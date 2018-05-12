// -*- coding: utf-8 -*-

#pragma once

#include "../common/range.cuh"



__inline__ __device__ int get_gid()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}


template<typename V>
__global__ void initialize_labels(V *labels, V n)
{
    for (auto i: grid_stride_range(n)) {
        // Vertex i is labeled i initially
        labels[i] = i;
    }
}



// Create an array (dst) of neighbor labels
template<typename V, typename E>
__global__ void gather_labels(const V *neighbors, const V *labels, V *dst, E m)
{
    int gid = get_gid();

    if (gid < m) {
        auto my_neighbor = neighbors[gid];
        auto my_label = labels[my_neighbor];
        dst[gid] = my_label;
    }
}


// Create a flag array to indicate the index where the labels change
template<typename V, typename E>
__global__ void find_segments(V *adj_labels, E m, int *segments)
{
    int gid = get_gid();
    if (gid < m - 1) {
        if (adj_labels[gid] != adj_labels[gid + 1]) {
            segments[gid] = 1u;
        } else {
            segments[gid] = 0u;
        }
    }
}


// Complement the flag array for boundary cases
template<typename V, typename E>
__global__ void set_boundary_case(E *offsets, V n, int *segments)
{
    int gid = get_gid();
    if (gid < n) {
        segments[offsets[gid + 1] - 1] = 1u;
    }
}


template<typename V, typename E>
__global__ void scatter_indexes(V *adj_labels, E *offsets, V n, E m, int *segments, int *seg_offsets)
{
    int gid = get_gid();
    if (gid < m) {
        if (adj_labels[gid] != adj_labels[gid + 1]) {
            seg_offsets[adj_labels[gid]] = gid + 1;
        }
    }

    if (gid < n) {
        if (gid == 0) {
            segments[-1] = 0;
        }
        segments[gid] = adj_labels[offsets[gid + 1]];
    }
}


__global__ void compute_label_weights
(int *segments, int *limitp, int *label_weights)
{
    int limit = limitp[0];
    for (auto t: grid_stride_range(limit)) {
        label_weights[t] = t == 0 ? segments[t] : segments[t] - segments[t - 1];
    }
}


template<typename V>
__global__ void update_labels
(int *adj_labels, int *segments, int *label_index, V n, int *labels, int *counter)
{
    int gid = get_gid();
    __shared__ int s_count;
    if (threadIdx.x == 0) {
        s_count = 0;
    }
    __syncthreads();

    if (gid < n) {
        int i = label_index[gid];
        int b = segments[i];
        int lbl = adj_labels[b - 1];
        if (lbl != labels[gid]) {
            atomicAdd(&s_count, 1);
        }
        labels[gid] = lbl;
    }

    if (threadIdx.x == 0) {
        atomicAdd(counter, s_count);
    }
}




////////////////////////////////////////////////////////////////////////////////
// Load imbalance functions
__device__ __inline__ bool _update(int w1, int w2, int i1, int i2)
{
    // return ((w1 < w2) ||
    //         ((w1 == w2) &&
    //          ((i1 ^ i2) & 1) == 1));
    return (w1 <= w2);
}


__global__ void compute_max_labels
(int *segments, int *seg_offsets, int *adj_labels, int *labels, int n, int *counter)
{
    __shared__ int s_weights[128];
    __shared__ int s_indexes[128];

    int c = 0;

    for (int v = blockIdx.x; v < n; v += gridDim.x) {
        __syncthreads();

        int my_weight = 0u;
        int my_index = 0u;

        int left = v == 0 ? 0 : segments[v - 1];
        int right = segments[v];
        for (int j = left + threadIdx.x; j < right; j += blockDim.x) {
            if (j == 0) {
                my_weight = my_index = seg_offsets[0];
            } else {
                int w = seg_offsets[j] - seg_offsets[j - 1];
                if (_update(my_weight, w, my_index, seg_offsets[j])) {
                    my_weight = w;
                    my_index = seg_offsets[j];
                }
            }
        }
        s_weights[threadIdx.x] = my_weight;
        s_indexes[threadIdx.x] = my_index;
        __syncthreads();

        if (threadIdx.x < 32) {
#pragma unroll
            for (int i = 1; i < 4; ++i) {
                if (_update(my_weight, s_weights[32 * i + threadIdx.x],
                            my_index, s_indexes[32 * i + threadIdx.x])) {
                    s_weights[threadIdx.x] = (my_weight = s_weights[32 * i + threadIdx.x]);
                    s_indexes[threadIdx.x] = (my_index = s_indexes[32 * i + threadIdx.x]);
                }
            }

            {
                volatile int *vs_weights = s_weights;
                volatile int *vs_indexes = s_indexes;

#pragma unroll
                for (int stride = 16; stride > 0; stride >>= 1) {
                    if (threadIdx.x < stride) {
                        if (_update(vs_weights[threadIdx.x], vs_weights[threadIdx.x + stride],
                                    vs_indexes[threadIdx.x], vs_indexes[threadIdx.x + stride])) {
                            vs_weights[threadIdx.x] = vs_weights[threadIdx.x + stride];
                            vs_indexes[threadIdx.x] = vs_indexes[threadIdx.x + stride];
                        }
                    }
                }
            }
        }

        if (threadIdx.x == 0) {
            if (s_indexes[0] > 0 && s_indexes[0] < seg_offsets[segments[n - 1] - 1]) {
                if (labels[v] != adj_labels[s_indexes[0] - 1]) {
                    ++c;
                }
                labels[v] = adj_labels[s_indexes[0] - 1];
            }
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(counter, c);
    }
}
