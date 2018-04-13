// -*- coding: utf-8 -*-

#pragma once

#include "label_propagator.h"
#include "segmented_reduce.cuh"
#include "kernel.cuh"

#include "moderngpu.cuh"
#include "kernels/segmentedsort.cuh"


// Data-parallel primitives based label propagation
template<typename V, typename E>
class DPP: public LabelPropagator<V, E> {
public:
    using typename LabelPropagator<V, E>::GraphT;

    DPP(std::shared_ptr<GraphT> _G, int sr_buf=0, bool nalb=false);
    virtual ~DPP();

    std::pair<double, double> run(int niter);


protected:
    using LabelPropagator<V, E>::G;
    using LabelPropagator<V, E>::labels;

    virtual void preprocess() = 0;
    virtual int iterate(int i) = 0;
    virtual void postprocess() = 0;

    void perform_lp(int n, int m, int lbl_offset=0, cudaStream_t stream=0, int *pinned_hmem=nullptr);
    int get_count();

    void init_gmem(V n, int m);
    void free_gmem();

    // GPU memory
    int *d_neighbors;      // m
    int *d_offsets;        // n + 1
    int *d_labels;         // n
    int *d_adj_labels;     // m
    int *d_tmp1;           // m + 1
    int *d_tmp2;           // m + 1
    int *d_segments;       // n + 1
    int *d_label_weights;  // m
    int *d_tmp_labels;     // n
    int *d_counter;        // 1

    mgpu::ContextPtr context;  // Used for segmented sort and scan
    SegmentedReducer reducer;

    bool no_adj_labels_buf; // Don't allocate d_adj_labels if true
};


template<typename V, typename E>
DPP<V, E>::DPP(std::shared_ptr<GraphT> _G, int sr_buf, bool nalb)
    : LabelPropagator<V, E>(_G), context(mgpu::CreateCudaDeviceStream(0)),
    reducer(sr_buf > 0 ? sr_buf : _G->m, context->Stream()), no_adj_labels_buf(nalb)
{
    cudaHostRegister((void *) &G->neighbors[0], sizeof(V) * G->m, cudaHostRegisterMapped);
    cudaHostRegister((void *) &G->offsets[0], sizeof(E) * (G->n + 1), cudaHostRegisterMapped);
}


template<typename V, typename E>
DPP<V, E>::~DPP()
{
    cudaHostUnregister((void *) &G->neighbors[0]);
    cudaHostUnregister((void *) &G->offsets[0]);
}


template<typename V, typename E>
std::pair<double, double> DPP<V, E>::run(int niter)
{
    Timer t1, t2;
    t2.start();

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

    cudaMemcpy(labels.get(), d_labels, sizeof(V) * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    t1.stop();
    t2.stop();

    postprocess();

    return std::make_pair(t1.elapsed_time(), t2.elapsed_time());
}


template<typename V, typename E>
void DPP<V, E>::perform_lp(int n, int m, int lbl_offset, cudaStream_t stream, int *pinned_hmem)
{
    const int nthreads = 256;
    const int n_blocks = divup(n, nthreads);
    const int m_blocks = divup(m, nthreads);

    gather_labels<<<m_blocks, nthreads, 0, stream>>>(d_neighbors, d_labels, d_adj_labels, m);

    mgpu::SegSortKeysFromIndices(d_adj_labels, m, d_offsets + 1, n - 1, *context);

    find_segments<<<m_blocks, nthreads, 0, stream>>>(d_adj_labels, m, d_tmp1);
    set_boundary_case<<<n_blocks, nthreads, 0, stream>>>(d_offsets, n, d_tmp1);

    mgpu::ScanExc(d_tmp1, m + 1, *context);

    scatter_indexes<<<m_blocks, nthreads, 0, stream>>>(d_tmp1, d_offsets, n, m, d_segments + 1, d_tmp2);

    compute_label_weights<<<n_blocks, nthreads, 0, stream>>>
        (d_tmp2, d_segments + n, d_label_weights);

    int N;
    if (pinned_hmem != nullptr) {
        cudaMemcpyAsync(pinned_hmem, d_segments + n, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        N = *pinned_hmem;
    } else {
        cudaMemcpy(&N, d_segments + n, sizeof(int), cudaMemcpyDeviceToHost);
    }

    reducer.apply(d_label_weights, N, d_segments, n, d_tmp1, d_tmp_labels);

    update_labels<<<n_blocks, nthreads, 0, stream>>>
        (d_adj_labels, d_tmp2, d_tmp_labels, n, d_labels + lbl_offset, d_counter);
}


// Return the number of labels updated
template<typename V, typename E>
int DPP<V, E>::get_count()
{
    int counter;
    cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(d_counter, 0, sizeof(int));

    return counter;
}


template<typename V, typename E>
void DPP<V, E>::init_gmem
(V n, int m)
{
    cudaMalloc(&d_neighbors,      sizeof(int) * m);
    cudaMalloc(&d_offsets,        sizeof(int) * (n + 1));
    cudaMalloc(&d_labels,         sizeof(int) * n);
    if (!no_adj_labels_buf) {
        cudaMalloc(&d_adj_labels,     sizeof(int) * (m + 1));
    } else {
        d_adj_labels = d_neighbors;
    }
    cudaMalloc(&d_tmp1,           sizeof(int) * (m + 1));
    cudaMalloc(&d_tmp2,           sizeof(int) * (m + 1));
    cudaMalloc(&d_segments,       sizeof(int) * (n + 1));
    cudaMalloc(&d_label_weights,  sizeof(int) * m);
    cudaMalloc(&d_tmp_labels,     sizeof(int) * n);
    cudaMalloc(&d_counter,        sizeof(int) * 1);

    cudaMemset(d_counter, 0, sizeof(int));
}


template<typename V, typename E>
void DPP<V, E>::free_gmem()
{
    cudaFree(d_neighbors);
    cudaFree(d_offsets);
    cudaFree(d_labels);
    if (!no_adj_labels_buf) {
        cudaFree(d_adj_labels);
    }
    cudaFree(d_tmp1);
    cudaFree(d_tmp2);
    cudaFree(d_segments);
    cudaFree(d_label_weights);
    cudaFree(d_tmp_labels);
    cudaFree(d_counter);
}