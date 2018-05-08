// -*- coding: utf-8 -*-

#pragma once

#include <vector>

#include "graph.h"
#include "label_propagator.h"
#include "outofcore.cuh"

#include "nccl.h"


template<typename V, typename E, typename S>
// class MultiAsyncLP: public LabelPropagator<V, E>, public OutOfCore<V, E> {
class MultiAsyncLP: public LabelPropagator<V, E> {
public:
    using typename LabelPropagator<V, E>::GraphT;

    MultiAsyncLP(std::shared_ptr<GraphT> _G, int _ngpus, int p, int _bs)
        : LabelPropagator<V, E>(_G), ngpus(_ngpus), policy(p), bs(_bs) { }
    MultiAsyncLP(std::shared_ptr<GraphT> _G, int _ngpus, int bs)
        : MultiAsyncLP(_G, _ngpus, -1, bs) { }
    virtual ~MultiAsyncLP() = default;

    std::pair<double, double> run(int niter);

private:
    // Methods
    void preprocess();
    int iterate(int i, int gpu);
    void postprocess();

    // Attributes
    using LabelPropagator<V, E>::G;

    int ngpus;
    int policy;  // Used for LFHT variants
    int bs;

    S **propagators;

    ncclComm_t *comms;
};


template<typename V, typename E, typename S>
std::pair<double, double> MultiAsyncLP<V, E, S>::run(int niter)
{
    Timer t1, t2;
    t2.start();

    Timer t; t.start();

    preprocess();

    t.stop();
    printf("preprocess: %f\n", t.elapsed_time());

    t1.start();

    const auto n = G->n;
    const int nthreads = 128;
    const int n_blocks = divup(n, nthreads);

    int total_count = 0;

    #pragma omp parallel num_threads(ngpus)
    {
        int gpu = omp_get_thread_num();
        cudaSetDevice(gpu);

        if (gpu == 0) {
            V *h_neighbors;
            E *h_offsets;
            cudaHostGetDevicePointer((void **) &h_neighbors, (void *) &G->neighbors[0], 0);
            cudaHostGetDevicePointer((void **) &h_offsets, (void *) &G->offsets[0], 0);
            initialize_labels<<<n_blocks, nthreads>>>
                (propagators[gpu]->d_labels, n, h_neighbors, h_offsets);
            cudaDeviceSynchronize();
        }

        if (ngpus > 1) {
            ncclBcast((void *) (propagators[gpu]->d_labels), n, ncclInt, 0, comms[gpu], propagators[gpu]->stream1);
            cudaDeviceSynchronize();
        }

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

    cudaMemcpy(this->labels.get(), propagators[0]->d_labels, sizeof(V) * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    t1.stop();
    t2.stop();

    postprocess();

    return std::make_pair(t1.elapsed_time(), t2.elapsed_time());
}


template<typename V, typename E, typename S>
void MultiAsyncLP<V, E, S>::preprocess()
{
    propagators = new S*[ngpus];

    comms = (ncclComm_t *) malloc(sizeof(ncclComm_t) * ngpus);

    // ncclUniqueId id;
    // ncclGetUniqueId(&id);
    ncclCommInitAll(comms, ngpus, nullptr);

    #pragma omp parallel for num_threads(ngpus)
    for (int i = 0; i < ngpus; ++i) {
        cudaSetDevice(i);

        // ncclCommInitRank(&comms[i], ngpus, id, i);

        #pragma omp critical
        if (policy >= 0) {
            propagators[i] = new S(G, policy, bs, i);
        } else {
            // propagators[i] = new S(G, bs, i);
        }
        propagators[i]->preprocess();
    }
}


template<typename V, typename E, typename S>
int MultiAsyncLP<V, E, S>::iterate(int i, int gpu)
{
    S *P = propagators[gpu];

    if (i == 0) {
        // The first batch
        P->transfer_batch(gpu, P->d_neighbors, P->d_offsets, P->stream1);
        cudaDeviceSynchronize();
    }

    int nbatches = P->get_num_batches();
    // for (auto j: range(gpu, nbatches, ngpus)) {
    for (auto a: range(0, nbatches, ngpus)) {
        int j = a + gpu;

        if (j + ngpus < nbatches) {
            P->transfer_batch(j + ngpus, P->d_neighbors_buf, P->d_offsets_buf, P->stream2);
        } else {
            P->transfer_batch(gpu, P->d_neighbors_buf, P->d_offsets_buf, P->stream2);
        }

        if (j < nbatches) {
            int batch_n = P->get_num_batch_vertices(j);
            int batch_m = P->get_num_batch_edges(j);
            P->perform_lp(batch_n, batch_m, P->bbs[j], &(P->h_norm_offsets[G->n + 1]), P->stream1);
        }

        for (auto g: range(ngpus)) {
            int b = a + g;
            if (b < nbatches) {
                int bn = P->get_num_batch_vertices(b);
                ncclBcast((void *) (P->d_labels + P->bbs[b]), bn, ncclInt, g, comms[gpu], P->stream1);
            }
        }

        cudaDeviceSynchronize();

        P->swap_buffers();
    }

    int count = P->get_count();
    return count;
}


template<typename V, typename E, typename S>
void MultiAsyncLP<V, E, S>::postprocess()
{
    #pragma omp parallel for num_threads(ngpus)
    for (int i = 0; i < ngpus; ++i) {
        cudaSetDevice(i);

        propagators[i]->postprocess();
        delete propagators[i];
    }

    for(int i = 0; i < ngpus; ++i) {
        ncclCommDestroy(comms[i]);
    }
    free(comms);

    delete[] propagators;
}
