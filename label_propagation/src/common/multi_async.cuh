// -*- coding: utf-8 -*-

#pragma once

#include <vector>

#include "graph.h"
#include "label_propagator.h"
#include "outofcore.cuh"


template<typename V, typename E, typename S>
class MultiAsyncLP: public LabelPropagator<V, E>, public OutOfCore<V, E> {
public:
    using typename LabelPropagator<V, E>::GraphT;

    MultiAsyncLP(std::shared_ptr<GraphT> _G, int _ngpus, int p, int bs)
        : LabelPropagator<V, E>(_G), OutOfCore<V, E>(_G, 1), ngpus(_ngpus),
          policy(p), h_bufs(ngpus), d_neighbors_bufs(ngpus),
          d_offsets_bufs(ngpus), main_streams(ngpus), sub_streams(ngpus) { }
    MultiAsyncLP(std::shared_ptr<GraphT> _G, int _ngpus, int bs)
        : MultiAsyncLP(_G, _ngpus, -1, bs) { }
    virtual ~MultiAsyncLP() = default;

    std::pair<double, double> run(int niter);

private:
    // Methods
    void preprocess();
    int iterate(int i, int gpu);
    void postprocess();

    void transfer_batch(int, std::vector<int> &, int *, int *, cudaStream_t);
    void swap_buffers(int gpu);

    // Attributes
    using LabelPropagator<V, E>::G;

    int ngpus;
    int policy;  // Used for LFHT variants

    std::vector<S> propagators;

    std::vector<int *> d_neighbors_bufs;
    std::vector<int *> d_offsets_bufs;
    std::vector<std::vector<int>> h_bufs;

    std::vector<cudaStream_t> main_streams;
    std::vector<cudaStream_t> sub_streams;

};


template<typename V, typename E, typename S>
std::pair<double, double> MultiAsyncLP<V, E, S>::run(int niter)
{
    Timer t1, t2;
    t2.start();

    cudaHostRegister((void *) &G->neighbors[0], sizeof(V) * G->m, cudaHostRegisterMapped);
    cudaHostRegister((void *) &G->offsets[0], sizeof(E) * (G->n + 1), cudaHostRegisterMapped);

    preprocess();

    t1.start();

    const auto n = G->n;
    const int nthreads = 128;
    const int n_blocks = divup(n, nthreads);

    int total_count = 0;

    #pragma omp parallel num_threads(ngpus)
    {
        int gpu = omp_get_thread_num();
        cudaSetDevice(gpu);

        V *h_neighbors;
        E *h_offsets;
        cudaHostGetDevicePointer((void **) &h_neighbors, (void *) &G->neighbors[0], 0);
        cudaHostGetDevicePointer((void **) &h_offsets, (void *) &G->offsets[0], 0);
        initialize_labels<<<n_blocks, nthreads>>>
            (propagators[i].d_labels, n, h_neighbors, h_offsets);

        // Main loop
        for (auto i: range(niter)) {
            Timer t_iter; t_iter.start();

            int count = iterate(i, gpu);
            #pragma omp atomic
            total_count += count;

            #pragma omp barrier

            if (gpu == 0) {
                t_iter.stop();
                printf("%d: %f (s), updated %d\n",
                       i + 1, t_iter.elapsed_time(), total_count);
            }

            if (total_count <= n * 1e-5) {
                break;
            }

            #pragma omp barrier

            if (gpu == 0) {
                total_count = 0;
            }
        }
    }

    // cudaMemcpy(this->labels.get(), d_labels, sizeof(V) * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    t1.stop();
    t2.stop();

    cudaHostUnregister((void *) &G->neighbors[0]);
    cudaHostUnregister((void *) &G->offsets[0]);

    postprocess();

    return std::make_pair(t1.elapsed_time(), t2.elapsed_time());
}


template<typename V, typename E, typename S>
void MultiAsyncLP<V, E, S>::preprocess()
{
    this->compute_batch_boundaries();

    for (int i = 0; i < ngpus; ++i) {
        cudaSetDevice(i);

        cudaMalloc(&&d_neighbors_bufs[i], sizeof(int) * this->B);
        cudaMalloc(&&d_offsets_bufs[i], sizeof(int) * G->n);
        h_bufs[i].resize(G->n + 2);
        cudaHostRegister(&(h_bufs[i][0]), sizeof(int) * (G->n + 2),
                         cudaHostRegisterMapped);

        if (policy >= 0) {
            propagators.emplace_back(G, policy, i);
        } else {
            propagators.emplace_back(G, i);
        }
        propagators.init_gmem(G->n, this->B);

        main_streams[i] = propagators[i].context->Stream();
        cudaStreamCreateWithFlags(&sub_streams[i], cudaStreamNonBlocking);
    }
}


template<typename V, typename E, typename S>
int MultiAsyncLP<V, E, S>::iterate(int i, int gpu)
{
    S &P = propagators[gpu];

    if (i == 0) {
        // The first batch
        transfer_batch(gpu, h_bufs[gpu], P.d_neighbors, P.d_offsets, main_streams[gpu]);
        cudaDeviceSynchronize();
    }

    int nbatches = this->get_num_batches();
    for (auto j: range(gpu, nbatches, ngpus)) {
        int batch_n = this->get_num_batch_vertices(j);
        int batch_m = this->get_num_batch_edges(j);

        if (j + ngpus < nbatches) {
            transfer_batch(j + ngpus, h_bufs[gpu], d_neighbors_bufs[gpu],
                           d_offsets_bufs[gpu], sub_streams[gpu]);
        } else {
            transfer_batch(gpu, h_bufs[gpu], d_neighbors_bufs[gpu],
                           d_offsets_bufs[gpu], sub_streams[gpu]);
        }

        P.perform_lp(batch_n, batch_m, this->bbs[j],
                     &h_bufs[gpu][G->n + 1], main_streams[gpu]);
        cudaDeviceSynchronize();

        swap_buffers(gpu);
    }

    int count = P.get_count();
    return count;
}


template<typename V, typename E, typename S>
void MultiAsyncLP<V, E, S>::postprocess()
{
    for (int i = 0; i < ngpus; ++i) {
        cudaSetDevice(i);

        cudaFree(d_neighbors_bufs[i]);
        cudaFree(d_offsets_bufs[i]);

        cudaHostUnregister(&h_bufs[i][0]);

        propagators[i].free_gmem();

        // cudaStreamDestroy(stream1);
        cudaStreamDestroy(sub_streams[i]);
    }
}


template<typename V, typename E, typename S>
void MultiAsyncLP<V, E, S>::swap_buffers(int gpu)
{
    S &P = propagators[gpu];
    std::swap(P.d_neighbors, d_neighbors_bufs[gpu]);
    P.d_adj_labels = P.d_neighbors;
    std::swap(P.d_offsets, d_offsets_bufs[gpu]);
}
