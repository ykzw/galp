// -*- coding: utf-8 -*-

#pragma once

#include <cmath>
#include <vector>

#include "graph.h"
#include "label_propagator.h"

#include "nccl.h"


template<typename V, typename E>
class MultiInCoreLP: public LabelPropagator<V, E> {
public:
    using typename LabelPropagator<V, E>::GraphT;

    MultiInCoreLP(std::shared_ptr<GraphT> _G, int _ngpus, int p)
        : LabelPropagator<V, E>(_G), ngpus(_ngpus), policy(p), d_neighbors(ngpus),
          d_offsets(ngpus), d_labels(ngpus), d_counter(ngpus), d_tables(ngpus), streams(ngpus) { }
    MultiInCoreLP(std::shared_ptr<GraphT> _G, int _ngpus)
        : MultiInCoreLP(_G, _ngpus, -1) { }
    virtual ~MultiInCoreLP() = default;

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

    std::vector<int> gpu_boundaries;

    std::vector<int *> d_neighbors;    // m
    std::vector<int *> d_offsets;      // n + 1
    std::vector<int *> d_labels;       // n
    std::vector<int *> d_counter;      // 1
    std::vector<GlobalHT> d_tables;    // m * 2
    std::vector<cudaStream_t> streams;

    double scale_factor;
    ncclComm_t *comms;
};


template<typename V, typename E>
std::pair<double, double> MultiInCoreLP<V, E>::run(int niter)
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

        initialize_labels<<<n_blocks, nthreads>>>(d_labels[gpu], n);

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
                scale_factor = 1.0;
            }
        }
        cudaDeviceSynchronize();
    }

    cudaMemcpy(this->labels.get(), d_labels[0], sizeof(V) * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    t1.stop();
    t2.stop();

    postprocess();

    return std::make_pair(t1.elapsed_time(), t2.elapsed_time());
}


template<typename V, typename E>
void MultiInCoreLP<V, E>::preprocess()
{
    {
        gpu_boundaries.push_back(0);
        E offset = G->m / ngpus;
        while (offset < G->m) {
            V left = 0;
            V right = G->n + 1;
            while (left < right) {
                V mid = (left + right) / 2;
                if (G->offsets[mid] < offset) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            gpu_boundaries.push_back(left);
            offset = G->offsets[left] + G->m / ngpus;
        }
        gpu_boundaries.push_back(G->n);
    }

    comms = (ncclComm_t *) malloc(sizeof(ncclComm_t) * ngpus);

    // ncclUniqueId id;
    // ncclGetUniqueId(&id);
    ncclCommInitAll(comms, ngpus, nullptr);

    scale_factor = 1.1;

    #pragma omp parallel for num_threads(ngpus)
    for (int i = 0; i < ngpus; ++i) {
        cudaSetDevice(i);

        // ncclCommInitRank(&comms[i], ngpus, id, i);

        int bn = gpu_boundaries[i + 1] - gpu_boundaries[i];
        int bm = G->offsets[gpu_boundaries[i + 1]] - G->offsets[gpu_boundaries[i]];

        cudaMalloc(&d_neighbors[i], sizeof(int) * bm);
        cudaMalloc(&d_offsets[i], sizeof(int) * (bn + 1));
        cudaMalloc(&d_labels[i], sizeof(int) * G->n);
        cudaMalloc(&d_counter[i], sizeof(int));
        int tm = bm * scale_factor;
        printf("%ld\n", sizeof(int) * ((long) bm + bn + G->n + tm + tm));
        cudaMalloc(&d_tables[i].keys, sizeof(int) * tm);
        cudaMalloc(&d_tables[i].vals, sizeof(int) * tm);
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

        cudaMemcpyAsync(d_neighbors[i], &G->neighbors[G->offsets[gpu_boundaries[i]]],
                        sizeof(int) * bm, cudaMemcpyHostToDevice, streams[i]);

        std::vector<int> tmp_offsets(bn + 1);
        for (auto j: range(gpu_boundaries[i], gpu_boundaries[i + 1] + 1)) {
            tmp_offsets[j - gpu_boundaries[i]] = G->offsets[j] - G->offsets[gpu_boundaries[i]];
        }
        cudaMemcpy(d_offsets[i], &tmp_offsets[0], sizeof(int) * (bn + 1), cudaMemcpyHostToDevice);
    }
}


template<typename V, typename E>
int MultiInCoreLP<V, E>::iterate(int i, int gpu)
{
    int bn = gpu_boundaries[gpu + 1] - gpu_boundaries[gpu];
    int bm = G->offsets[gpu_boundaries[gpu + 1]] - G->offsets[gpu_boundaries[gpu]];

    int tm = bm * scale_factor;
    cudaMemsetAsync(d_tables[gpu].keys, 0, sizeof(uint32_t) * tm, streams[gpu]);
    cudaMemsetAsync(d_tables[gpu].vals, 0, sizeof(uint32_t) * tm, streams[gpu]);

    const int nt = 64;
    update_lockfree<nt><<<bn, nt, 0, streams[gpu]>>>
        (d_neighbors[gpu], d_offsets[gpu], d_labels[gpu], d_tables[gpu], d_counter[gpu], gpu_boundaries[gpu], scale_factor);
    cudaDeviceSynchronize();

    if (ngpus > 1) {
        for (auto g: range(ngpus)) {
            int gn = gpu_boundaries[g + 1] - gpu_boundaries[g];
            ncclBcast((void *) (d_labels[gpu] + gpu_boundaries[g]), gn, ncclInt, g, comms[gpu], streams[gpu]);
        }
    }

    int count;
    cudaMemcpy(&count, d_counter[gpu], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(d_counter[gpu], 0, sizeof(int));
    return count;
}


template<typename V, typename E>
void MultiInCoreLP<V, E>::postprocess()
{
    #pragma omp parallel for num_threads(ngpus)
    for (int i = 0; i < ngpus; ++i) {
        cudaSetDevice(i);

        cudaFree(d_neighbors[i]);
        cudaFree(d_offsets[i]);
        cudaFree(d_labels[i]);
        cudaFree(d_counter[i]);
        cudaFree(d_tables[i].keys);
        cudaFree(d_tables[i].vals);
        cudaStreamDestroy(streams[i]);

        ncclCommDestroy(comms[i]);
    }
    free(comms);
}
