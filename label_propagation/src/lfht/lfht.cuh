// -*- coding: utf-8 -*-

#pragma once

#include "graph.h"
#include "label_propagator.h"
#include "hashtable.cuh"
#include "kernel.cuh"
#include "../common/segmented_reduce.cuh"

template<typename V, typename E, typename S>
class MultiAsyncLP;


template<typename V, typename E>
class LFHTBase: public LabelPropagator<V, E> {
public:
    friend MultiAsyncLP<V, E, LFHTBase<V, E>>;

    using GraphT = CSRGraph<V, E>;

    LFHTBase(std::shared_ptr<GraphT> _G, int p, bool nalb=true)
        : LabelPropagator<V, E>(_G), policy(p), context(mgpu::CreateCudaDeviceStream(0)), no_adj_labels_buf(nalb) { }
    LFHTBase(std::shared_ptr<GraphT> _G, int p, int gpu)
        : LabelPropagator<V, E>(_G), policy(p), context(mgpu::CreateCudaDeviceStream(gpu)), no_adj_labels_buf(true) { }
    virtual ~LFHTBase() = default;

    std::pair<double, double> run(int niter);


protected:
    // Methods
    virtual void preprocess() = 0;
    virtual int iterate(int i) = 0;
    virtual void postprocess() = 0;

    void perform_lp(int n, int m, int v_offset=0, int *pinned_hmem=nullptr, cudaStream_t stream=0);
    int get_count();

    void init_gmem(V n, int m);
    void free_gmem();

    // Attributes
    using LabelPropagator<V, E>::G;

    int policy;

    int *d_neighbors;    // m
    int *d_offsets;      // n + 1
    int *d_labels;       // n
    int *d_adj_labels;   // m
    int *d_counter;      // 1
    GlobalHT d_tables;   // m * 2

    mgpu::ContextPtr context;  // Used for scan
    SegmentedReducer *reducer;

    bool no_adj_labels_buf; // Don't allocate d_adj_labels if true

    // Used for load-balanced assignments
    int *d_num_blocks;     // n + 1
    int2 *d_assignments;   // m
    int *d_max_counts;     // m + 1

    // Used for a naive strategy
    int *d_tmp;          // n
    int *d_tmp_labels;   // n
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


template<typename V, typename E>
void LFHTBase<V, E>::perform_lp(int n, int m, int v_offset, int *pinned_hmem, cudaStream_t stream)
{
    cudaMemsetAsync(d_tables.keys, 0, sizeof(uint32_t) * m, stream);
    cudaMemsetAsync(d_tables.vals, 0, sizeof(uint32_t) * m, stream);

    const int nthreads = 256;
    const int n_blocks = divup(n, nthreads);
    const int m_blocks = divup(m, nthreads);

    const int nt = 32;
    switch (policy) {
    case 0:  // Naive
        gather_labels<<<m_blocks, nthreads, 0, stream>>>
            (d_neighbors, d_labels, d_adj_labels, m);
        count_lockfree<<<n, nt, 0, stream>>>
            (d_adj_labels, d_offsets, d_tables);
        reducer->apply((int *) d_tables.vals, m, d_offsets, n, d_tmp, d_tmp_labels);
        update_labels<<<n_blocks, nthreads, 0, stream>>>
            (d_tables.keys, d_tmp_labels, n, d_labels + v_offset, d_counter);
        break;

    case 1:  // Kernel fusion
        update_lockfree<nt><<<n, nt, 0, stream>>>
            (d_neighbors, d_offsets, d_labels, d_tables, d_counter, v_offset);
        break;

    case 2:  // Kernel fusion and shared memory
        update_lockfree_smem<nt, nt * 7><<<n, nt, 0, stream>>>
            (d_neighbors, d_offsets, d_labels, d_tables, d_counter, v_offset);
        break;

    case 3:// Kernel fusion, shared memory, and load balancing
        const int ts = nt * 3;
        cudaMemsetAsync(d_max_counts, 0, sizeof(int) * n, stream);
        compute_num_blocks<ts><<<n_blocks, nthreads, 0, stream>>>(d_offsets, n, d_num_blocks);
        mgpu::ScanExc(d_num_blocks, n + 1, *context);
        int nb;
        if (pinned_hmem != nullptr) {
            cudaMemcpyAsync(pinned_hmem, d_num_blocks + n, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            nb = *pinned_hmem;
        } else {
            cudaMemcpyAsync(&nb, d_num_blocks + n, sizeof(int), cudaMemcpyDeviceToHost, stream);
        }
        assign_blocks<ts><<<32, 256, 0, stream>>>(d_num_blocks, n, d_assignments);
        update_lockfree_smem_lb<nt, ts><<<nb, nt, 0, stream>>>
            (d_neighbors, d_offsets, d_labels, d_assignments, d_tables, d_max_counts, d_counter, v_offset);
        break;

    }
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

    switch (policy) {
    case 0:
        reducer = new SegmentedReducer(m, context->Stream());
        cudaMalloc(&d_tmp, sizeof(int) * n);
        cudaMalloc(&d_tmp_labels, sizeof(int) * n);
        break;
    case 3:
        cudaMalloc(&d_num_blocks, sizeof(int) * (n + 1));
        cudaMalloc(&d_assignments, sizeof(int2) * m);
        cudaMalloc(&d_max_counts, sizeof(int) * m);
        break;
    }
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

    switch (policy) {
    case 0:
        delete reducer;
        cudaFree(d_tmp);
        cudaFree(d_tmp_labels);
        break;
    case 3:
        cudaFree(d_num_blocks);
        cudaFree(d_assignments);
        cudaFree(d_max_counts);
        break;
    }

}
