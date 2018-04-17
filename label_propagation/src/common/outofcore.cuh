// -*- coding: utf-8 -*-

#pragma once

#include "graph.h"


// Out-of-core in the sense that the data does not fit into the GPU, but it fits into the CPU main memory
template<typename V, typename E>
class OutOfCore {
public:
    using GraphT = CSRGraph<V, E>;

    OutOfCore(std::shared_ptr<GraphT> _G, int bs): G(_G), h_norm_offsets(G->n + 2), B(1 << bs) {
        cudaHostRegister((void *) &h_norm_offsets[0], sizeof(int) * h_norm_offsets.size(), cudaHostRegisterMapped);
    }
    ~OutOfCore() {
        cudaHostUnregister((void *) &h_norm_offsets[0]);
    }

    void compute_batch_boundaries();
    void transfer_batch(int i, int *d_neighbors, int *d_offsets, cudaStream_t stream=0);
    void transfer_next_batch(int i, int *d_neighbors, int *d_offsets, cudaStream_t stream=0);

    int get_num_batches() { return bbs.size() - 1; }
    V get_num_batch_vertices(int i) { return bbs[i + 1] - bbs[i]; }
    E get_num_batch_edges(int i) { return G->offsets[bbs[i + 1]] - G->offsets[bbs[i]]; }

    std::shared_ptr<GraphT> G;

    std::vector<int> bbs;  // Batch BoundarieS
    std::vector<int> h_norm_offsets;

    int B;  // Buffer size
};


template<typename V, typename E>
void OutOfCore<V, E>::compute_batch_boundaries()
{
    bbs.push_back(0);

    E batch_offset = B;
    while (batch_offset < G->m) {
        V left = 0;
        V right = G->n + 1;
        while (left < right) {
            V mid = (left + right) / 2;
            if (G->offsets[mid] < batch_offset) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        bbs.push_back(left - 1);
        batch_offset = G->offsets[left - 1] + B;
    }

    bbs.push_back(G->n);
}


template<typename V, typename E>
void OutOfCore<V, E>::transfer_batch
(int i, int *d_neighbors, int *d_offsets, cudaStream_t stream)
{
    auto begin = bbs[i];
    auto end = bbs[i + 1];

    int v_offset = begin;
    int batch_n = end - v_offset;
    E e_offset = G->offsets[v_offset];
    E batch_m = G->offsets[end] - e_offset;
    cudaMemcpyAsync(d_neighbors, &G->neighbors[G->offsets[begin]],
                    sizeof(int) * batch_m, cudaMemcpyHostToDevice, stream);

    for (int k = v_offset; k <= end; ++k) {
        h_norm_offsets[k - v_offset] = (G->offsets[k] - e_offset);
    }
    cudaMemcpyAsync(d_offsets, &h_norm_offsets[0], sizeof(int) * (batch_n + 1),
                    cudaMemcpyHostToDevice, stream);
}


template<typename V, typename E>
void OutOfCore<V, E>::transfer_next_batch
(int i, int *d_neighbors, int *d_offsets, cudaStream_t stream)
{
    const int nbatches = bbs.size() - 1;
    if (i < nbatches - 1) {
        // The next batch at a current iteration
        transfer_batch(i + 1, d_neighbors, d_offsets, stream);
    } else {
        // The first batch at the next iteration
        transfer_batch(0, d_neighbors, d_offsets, stream);
    }
}
