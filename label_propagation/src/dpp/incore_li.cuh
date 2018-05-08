// -*- coding: utf-8 -*-

#pragma once

#include "dpp.cuh"
#include "cub/cub.cuh"


// GPU in-core with load imbalance, data-parallel primitives based label propagation
template<typename V, typename E>
class InCoreLIDPP: public DPPBase<V, E> {
public:
    using typename LabelPropagator<V, E>::GraphT;

    InCoreLIDPP(std::shared_ptr<GraphT> _G): DPPBase<V, E>(_G, false) { }
    virtual ~InCoreLIDPP() = default;


private:
    // Methods
    void preprocess();
    int iterate(int i);
    void postprocess();

    void transfer_data();

    // Attributes
    using LabelPropagator<V, E>::G;  // To avoid many "this->"es

};



template<typename V, typename E>
void InCoreLIDPP<V, E>::preprocess()
{
    this->init_gmem(G->n, G->m);
    transfer_data();
}


template<typename V, typename E>
int InCoreLIDPP<V, E>::iterate(int i)
{
    const int n = G->n;
    const int m = G->m;

    // Used by CUB radix sort
    static void *d_temp_storage = NULL;
    static size_t temp_storage_bytes = 0;

    const int nthreads = 128;
    const int n_blocks = divup(n, nthreads);
    const int m_blocks = divup(m, nthreads);

    gather_labels<<<m_blocks, nthreads>>>(this->d_neighbors, this->d_labels, this->d_adj_labels, m);

    if (temp_storage_bytes == 0){
        cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes, this->d_adj_labels, this->d_adj_labels,
            m, n, this->d_offsets, this->d_offsets + 1
        );

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }
    // Run sorting operation
    // *Load imbalanced*
    cub::DeviceSegmentedRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, this->d_adj_labels, this->d_adj_labels,
        m, n, this->d_offsets, this->d_offsets + 1
    );

    find_segments<<<m_blocks, nthreads>>>(this->d_adj_labels, m, this->d_tmp1);
    set_boundary_case<<<n_blocks, nthreads>>>(this->d_offsets, n, this->d_tmp1);

    mgpu::ScanExc(this->d_tmp1, G->m + 1, *this->context);

    scatter_indexes<<<m_blocks, nthreads>>>(this->d_tmp1, this->d_offsets, n, m, this->d_segments + 1, this->d_tmp2);

    // *Load imbalanced*
    compute_max_labels<<<n_blocks, nthreads>>>
        (this->d_segments + 1, this->d_tmp2, this->d_adj_labels, this->d_labels, n, this->d_counter);

    int count = this->get_count();
    return count;
}


template<typename V, typename E>
void InCoreLIDPP<V, E>::postprocess()
{
    this->free_gmem();
}


template<typename V, typename E>
void InCoreLIDPP<V, E>::transfer_data()
{
    cudaMemcpy(this->d_neighbors, &G->neighbors[0], sizeof(V) * G->m, cudaMemcpyHostToDevice);

    if (sizeof(E) > 4) {
        // Use 32-bit integers
        std::vector<int> tmp_offsets(G->n + 1);
        for (auto i: range(G->n + 1)) {
            tmp_offsets[i] = G->offsets[i];
        }
        cudaMemcpy(this->d_offsets, &tmp_offsets[0], sizeof(int) * (G->n + 1), cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(this->d_offsets, &G->offsets[0], sizeof(E) * (G->n + 1), cudaMemcpyHostToDevice);
    }
}