// -*- coding: utf-8 -*-

#pragma once

#include "graph.h"
#include "outofcore.cuh"


template<typename V, typename E, typename S>
class MultiAsyncLP;

// template<typename V, typename E, typename S>
// class MultiInCoreLP;


// GPU out-of-core with overlap, data-parallel primitives based label propagation
template<typename V, typename E, typename S>
class AsyncLP: public S, public OutOfCore<V, E> {
public:
    using typename LabelPropagator<V, E>::GraphT;

    AsyncLP(std::shared_ptr<GraphT> _G, int bs)
        : S(_G), OutOfCore<V, E>(_G, bs) { }
    AsyncLP(std::shared_ptr<GraphT> _G, int p, int bs)
        : S(_G, p), OutOfCore<V, E>(_G, bs) { }
    AsyncLP(std::shared_ptr<GraphT> _G, int p, int bs, int gpu)
        : S(_G, p, gpu), OutOfCore<V, E>(_G, bs) { }
    virtual ~AsyncLP() = default;

    friend MultiAsyncLP<V, E, AsyncLP<V, E, S>>;
    // friend MultiInCoreLP<V, E, AsyncLP<V, E, S>>;

private:
    // Methods
    void preprocess();
    int iterate(int i);
    void postprocess();

    void swap_buffers();

    void init_gmem(int n, int B);
    void free_gmem();

    // Attributes
    using LabelPropagator<V, E>::G;

    // For double buffering
    int *d_neighbors_buf;  // B
    int *d_offsets_buf;    // n + 1

    cudaStream_t stream1;
    cudaStream_t stream2;

};


template<typename V, typename E, typename S>
void AsyncLP<V, E, S>::preprocess()
{
    init_gmem(G->n, this->B);

    this->compute_batch_boundaries();

    stream1 = this->context->Stream();
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
}


template<typename V, typename E, typename S>
int AsyncLP<V, E, S>::iterate(int i)
{
    if (i == 0) {
        // The first batch
        this->transfer_batch(0, this->d_neighbors, this->d_offsets, stream1);
        cudaDeviceSynchronize();
    }

    int nbatches = this->get_num_batches();
    for (auto j: range(nbatches)) {
        int batch_n = this->get_num_batch_vertices(j);
        int batch_m = this->get_num_batch_edges(j);

        this->transfer_next_batch(j, d_neighbors_buf, d_offsets_buf, stream2);

        this->perform_lp(batch_n, batch_m, this->bbs[j], &this->h_norm_offsets[G->n + 1], stream1);
        cudaDeviceSynchronize();

        swap_buffers();
    }

    int count = this->get_count();
    return count;
}


template<typename V, typename E, typename S>
void AsyncLP<V, E, S>::postprocess()
{
    free_gmem();

    // cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}


template<typename V, typename E, typename S>
void AsyncLP<V, E, S>::swap_buffers()
{
    std::swap(this->d_neighbors, d_neighbors_buf);
    this->d_adj_labels = this->d_neighbors;
    std::swap(this->d_offsets, d_offsets_buf);
}


template<typename V, typename E, typename S>
void AsyncLP<V, E, S>::init_gmem(int n, int B)
{
    cudaMalloc(&d_neighbors_buf, sizeof(int) * B);
    cudaMalloc(&d_offsets_buf, sizeof(int) * (n + 1));

    S::init_gmem(n, B);
}


template<typename V, typename E, typename S>
void AsyncLP<V, E, S>::free_gmem()
{
    cudaFree(d_neighbors_buf);
    cudaFree(d_offsets_buf);

    S::free_gmem();
}