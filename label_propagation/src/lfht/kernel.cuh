// -*- coding: utf-8 -*-

#pragma once

#include "../common/range.cuh"


// Lock-free hash tables on global memory
// - Just perform counting
// - Block per vertex
__global__ void count_lockfree
(int *adj_labels, int *offsets, GlobalHT g_tables)
{
    const int v = blockIdx.x;
    const int begin = offsets[v];
    const int end = offsets[v + 1];
    for (auto i: block_stride_range(begin, end)) {
        auto key = adj_labels[i] + 1;
        g_tables.increment(begin, end, key);
    }
}

__global__ void update_labels
(uint32_t *keys, int *label_index, int n, int *labels, int *counter)
{
    int gid = get_gid();
    __shared__ int s_count;
    if (threadIdx.x == 0) {
        s_count = 0;
    }
    __syncthreads();

    if (gid < n) {
        int label = keys[label_index[gid]] - 1;
        if (label != labels[gid]) {
            atomicAdd(&s_count, 1);
        }
        labels[gid] = label;
    }

    if (threadIdx.x == 0) {
        atomicAdd(counter, s_count);
    }
}


// Kernel fusion of gather, count, and update
// - Perform the entire iteration by one kernel
// - Block per vertex
template<int NT>
__global__ void update_lockfree
(int *neighbors, int *offsets, int *labels, GlobalHT g_tables, int *counter, int v_offset=0)
{
    __shared__ int s_max_keys[NT];
    __shared__ int s_max_counts[NT];

    const int v = blockIdx.x;
    const int begin = offsets[v];
    const int end = offsets[v + 1];
    int my_max_count = 0;
    int my_max_key = 0;
    for (auto i: block_stride_range(begin, end)) {
        // Keys are >= 1; labels are >= 0
        auto key = labels[neighbors[i]] + 1;
        auto c = g_tables.increment(begin, end, key);
        if (c > my_max_count) {
            my_max_key = key;
            my_max_count = c;
        }
    }
    s_max_keys[threadIdx.x] = my_max_key;
    s_max_counts[threadIdx.x] = my_max_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < NT; ++i) {  // Naive reduction
            if (s_max_counts[i] > my_max_count) {
                my_max_key = s_max_keys[i];
                my_max_count = s_max_counts[i];
            }
        }
        if (labels[v + v_offset] != my_max_key - 1) {
            ++(*counter);
            labels[v + v_offset] = my_max_key - 1;
        }
    }
}


template<int SC>
__device__ void flush_s2g
(GlobalHT &g_tables, int begin, int end,
 SharedHT<SC> &s_table, int &my_max_key, int &my_max_count)
{
    // Flush s_table to g_tables
    for (auto i: block_stride_range(SC)) {
        auto key = s_table.keys[i];
        auto count = s_table.vals[i];
        if (key > 0) {
            auto c = g_tables.increment(begin, end, key, count);
            if (c > my_max_count) {
                my_max_key = key;
                my_max_count = c;
            }
        }
    }
    __syncthreads();
    s_table.clear();
    __syncthreads();
}


// Kernel fusion and shared memory hash table
// - TS elements are first aggregated on the shared memory,
//   and then flush it to the global hash tables
// - Still block per vertex
template<int NT, int TS>
__global__ void update_lockfree_smem
(int *neighbors, int *offsets, int *labels, GlobalHT g_tables, int *counter, int v_offset=0)
{
    constexpr int SC = TS + NT;  // Capacity of the smem hash table
    __shared__ SharedHT<SC> s_table;
    s_table.clear();
    __syncthreads();

    __shared__ int s_max_keys[NT];
    __shared__ int s_max_counts[NT];

    const int v = blockIdx.x;
    const int begin = offsets[v];
    const int end = offsets[v + 1];
    int my_max_key = 0;
    int my_max_count = 0;
    for (int i = begin; i < end; i += NT) {
        int j = i + threadIdx.x;
        if (j < end) {
            auto key = labels[neighbors[j]] + 1;
            s_table.increment(key);
        }

        if (s_table.nitems >= SC - NT) {
            __syncthreads();
            flush_s2g(g_tables, begin, end, s_table, my_max_key, my_max_count);
        }
    }
    __syncthreads();
    if (s_table.nitems > 0) {
        flush_s2g(g_tables, begin, end, s_table, my_max_key, my_max_count);
    }
    s_max_keys[threadIdx.x] = my_max_key;
    s_max_counts[threadIdx.x] = my_max_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < NT; ++i) {
            if (s_max_counts[i] > my_max_count) {
                my_max_key = s_max_keys[i];
                my_max_count = s_max_counts[i];
            }
        }
        auto lbl = labels[v + v_offset];
        if (lbl != my_max_key - 1) {
            ++(*counter);
            labels[v + v_offset] = my_max_key - 1;
        }
    }
}



template<int TS>
__global__ void compute_num_blocks // per vertex
(int *offsets, int n, int *num_blocks)
{
    for (auto i: grid_stride_range(n)) {
        num_blocks[i] = (offsets[i + 1] - offsets[i] + TS - 1) / TS;
    }
}


template<int TS>
__global__ void assign_blocks
(int *num_blocks, int n, int2 *assignments)
{
    const int nb = num_blocks[n];
    for (auto b: grid_stride_range(nb)) {
        // Binary search
        int left = 0;
        int right = n;
        while (left < right) {
            int mid = (left + right) / 2;
            if (num_blocks[mid] <= b) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Assignment information is a pair of integers:
        // 1. Block b handles vertex *(left - 1)*.
        // 2. This block is the *(b - num_blocks[left - 1])*-th block among the ones handling the same vertex.
        assignments[b] = make_int2(left - 1, b - num_blocks[left - 1]);
    }
}


// Kernel fusion, shared memory hash table, and load balancing
// - Multiple blocks per vertex according to its degree
// - Several other kernels need to be called before this one
template<int NT, int TS>
__global__ void update_lockfree_smem_lb
(int *neighbors, int *offsets, int *labels, int2 *assignments,
 GlobalHT g_tables, int *max_counts, int *d_counter, int v_offset=0)
{
    constexpr int SC = TS + NT;  // capacity
    __shared__ SharedHT<SC> s_table;
    s_table.clear();
    __syncthreads();

    const int2 info = assignments[blockIdx.x];
    const int vertex = info.x;
    const int begin = offsets[vertex];
    const int end = offsets[vertex + 1];
    const int tile_offset = begin + TS * info.y;

    // Shared memory aggregation
    #pragma unroll
    for (int i = 0; i < (TS / NT); ++i) {
        int t = NT * i + threadIdx.x;
        int j = tile_offset + t;
        if (j < end) {
            uint32_t key = labels[neighbors[j]] + 1;
            s_table.increment(key);
        }
    }
    __syncthreads();

    __shared__ uint32_t s_max_key;
    __shared__ uint32_t s_max_count;
    if (threadIdx.x == 0) {
        s_max_key = 0;
        s_max_count = 0;
    }
    __syncthreads();

    uint32_t my_max_key = 0;
    uint32_t my_max_count = 0;

    // Flush to g_tables, and
    // Find the most frequent key within the tile
    #pragma unroll
    for (int i = 0; i < (SC / NT); ++i) {
        int j = NT * i + threadIdx.x;
        auto key = s_table.keys[j];
        auto count = s_table.vals[j];
        if (key > 0) {
            auto c = g_tables.increment(begin, end, key, count);
            if (my_max_count < c) {
                my_max_key = key;
                my_max_count = c;
            }
        }
    }
    auto m = atomicMax(&s_max_count, my_max_count);
    if (m < my_max_count && s_max_count == my_max_count) {
        s_max_key = my_max_key;
    }
    __syncthreads();

    // Try to update the label of the vertex a block handles
    if (threadIdx.x == 0) {
        auto ret = atomicMax(&max_counts[vertex], s_max_count);
        if (ret < s_max_count && max_counts[vertex] == s_max_count) {
            auto lbl = labels[vertex + v_offset];
            labels[vertex + v_offset] = s_max_key - 1;
            if (lbl != s_max_key - 1) {
                // May count redundantly
                ++(*d_counter);
            }
        }
    }
}
