// -*- coding: utf-8 -*-

#pragma once

#include "graph.h"
#include "label_propagator.h"
#include "hashtable.cuh"


template<typename V, typename E>
class LFHTBase: public LabelPropagator<V, E> {
public:
    using GraphT = CSRGraph<V, E>;

    LFHTBase(std::shared_ptr<GraphT> _G, bool nalb=false)
        : LabelPropagator<V, E>(_G), context(mgpu::CreateCudaDeviceStream(0)), no_adj_labels_buf(nalb) { }
    virtual ~LFHTBase() = default;

    std::pair<double, double> run(int niter);


protected:
    // Methods
    virtual void preprocess() = 0;
    virtual int iterate(int i) = 0;
    virtual void postprocess() = 0;

    int get_count();

    void init_gmem(V n, int m);
    void free_gmem();

    // Attributes
    using LabelPropagator<V, E>::G;

    int *d_neighbors;    // m
    int *d_offsets;      // n + 1
    int *d_labels;       // n
    int *d_adj_labels;   // m
    int *d_counter;      // 1
    GlobalHT d_tables;   // m * 2

    mgpu::ContextPtr context;  // Used for scan

    bool no_adj_labels_buf; // Don't allocate d_adj_labels if true
};


template<typename V, typename E>
std::pair<double, double> LFHTBase<V, E>::run(int niter)
{
    Timer t1, t2;
    t2.start();

    cudaHostRegister((void *) &G->neighbors[0], sizeof(V) * G->m, cudaHostRegisterMapped);
    cudaHostRegister((void *) &G->offsets[0], sizeof(E) * (G->n + 1), cudaHostRegisterMapped);

    preprocess();

    const auto n = G->n;
    const int nthreads = 256;
    const int n_blocks = divup(n, nthreads);
    // initialize_labels<<<n_blocks, nthreads>>>(d_labels, n);

    V *h_neighbors;
    E *h_offsets;
    cudaHostGetDevicePointer((void **) &h_neighbors, (void *) &G->neighbors[0], 0);
    cudaHostGetDevicePointer((void **) &h_offsets, (void *) &G->offsets[0], 0);
    initialize_labels<<<n_blocks, nthreads>>>(d_labels, n, h_neighbors, h_offsets);

    t1.start();
    // Main loop
    for (auto i: range(niter)) {
        Timer t_iter; t_iter.start();

        int count = iterate(i);

        t_iter.stop();
        printf("%d: %f (s), updated %d\n", i + 1, t_iter.elapsed_time(), count);

        if (count <= n * 1e-5) {
            break;
        }
    }

    cudaMemcpy(this->labels.get(), d_labels, sizeof(V) * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    t1.stop();
    t2.stop();

    cudaHostUnregister((void *) &G->neighbors[0]);
    cudaHostUnregister((void *) &G->offsets[0]);

    postprocess();

    return std::make_pair(t1.elapsed_time(), t2.elapsed_time());
}


// Return the number of labels updated
template<typename V, typename E>
int LFHTBase<V, E>::get_count()
{
    int counter;
    cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(d_counter, 0, sizeof(int));

    return counter;
}


template<typename V, typename E>
void LFHTBase<V, E>::init_gmem(V n, int m)
{
    cudaMalloc(&d_neighbors,      sizeof(int) * m);
    cudaMalloc(&d_offsets,        sizeof(int) * (n + 1));
    cudaMalloc(&d_labels,         sizeof(int) * n);
    if (!no_adj_labels_buf) {
        cudaMalloc(&d_adj_labels,     sizeof(int) * (m + 1));
    } else {
        d_adj_labels = d_neighbors;
    }
    cudaMalloc(&d_counter,        sizeof(int) * 1);

    cudaMemset(d_counter, 0, sizeof(int));

    cudaMalloc(&(d_tables.keys), sizeof(uint32_t) * m);
    cudaMalloc(&(d_tables.vals), sizeof(uint32_t) * m);
}


template<typename V, typename E>
void LFHTBase<V, E>::free_gmem()
{
    cudaFree(d_neighbors);
    cudaFree(d_offsets);
    cudaFree(d_labels);
    if (!no_adj_labels_buf) {
        cudaFree(d_adj_labels);
    }
    cudaFree(d_counter);

    cudaFree(d_tables.keys);
    cudaFree(d_tables.vals);
}
