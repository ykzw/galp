// -*- coding: utf-8 -*-

#pragma once

#include "lfht.cuh"
#include "kernel.cuh"
#include "../common/range.cuh"


// GPU in-core, lock-free hash table based label propagation
template<typename V, typename E>
class InCoreLFHT: public LFHTBase<V, E> {
public:
    using typename LabelPropagator<V, E>::GraphT;

    InCoreLFHT(std::shared_ptr<GraphT> _G): LFHTBase<V, E>(_G) { }
    virtual ~InCoreLFHT() = default;


protected:
    // Methods
    void preprocess();
    int iterate(int i);
    void postprocess();

    void transfer_data();

    // Attributes
    using LabelPropagator<V, E>::G;

    int *d_num_blocks;
    int2 *d_assignments;
};


template<typename V, typename E>
void InCoreLFHT<V, E>::preprocess()
{
    this->init_gmem(G->n, G->m);
    transfer_data();

    cudaMalloc((void **) &d_num_blocks, sizeof(int) * (G->n + 1));
    cudaMalloc((void **) &d_assignments, sizeof(int2) * G->m);
}


template<typename V, typename E>
int InCoreLFHT<V, E>::iterate(int i)
{
    cudaMemset(this->d_tables.keys, 0, sizeof(uint32_t) * G->m * 1);
    cudaMemset(this->d_tables.vals, 0, sizeof(uint32_t) * G->m * 1);

    const int nt = 32;

    const int n_blocks = divup(G->n, 256);
    const int m_blocks = divup(G->m, 256);

    // update_lockfree<nt><<<G->n, nt>>>
    //     (this->d_neighbors, this->d_offsets, this->d_labels, this->d_tables, this->d_counter);
    update_lockfree_smem<nt, nt * 7><<<G->n, nt>>>
        (this->d_neighbors, this->d_offsets, this->d_labels, this->d_tables, this->d_counter);

    // gather_labels<<<m_blocks, 256>>>(this->d_neighbors, this->d_labels, this->d_adj_labels, G->m);
    // count_lockfree<nt><<<G->n, nt>>>(this->d_adj_labels, this->d_offsets, this->d_labels, this->d_tables, this->d_counter);

    // NaiveS()(G->n, G->m, this->d_neighbors, this->d_offsets, this->d_labels,
    //         this->d_adj_labels, this->d_tables, this->d_counter);

    // count_lockfree2<nt><<<G->n, nt>>>(this->d_neighbors, this->d_offsets, this->d_labels, this->d_tables, this->d_counter);
    // FusedS()(G->n, G->m, this->d_neighbors, this->d_offsets, this->d_labels, this->d_tables, this->d_counter);

    // count_lockfree2_2<nt><<<G->n, nt>>>(this->d_neighbors, this->d_offsets, this->d_labels, this->d_tables, this->d_counter);
    // SharedFusedS()(G->n, G->m, this->d_neighbors, this->d_offsets, this->d_labels, this->d_tables, this->d_counter);


    // const int NT = 64;
    // const int TS = NT * 2;
    // cudaStream_t stream1 = 0;
    // compute_num_blocks<TS><<<n_blocks, 256, 0, stream1>>>(this->d_offsets, G->n, d_num_blocks);
    // mgpu::ScanExc(d_num_blocks, G->n + 1, *this->context);
    // int nb;
    // cudaMemcpy(&nb, d_num_blocks + G->n, sizeof(int), cudaMemcpyDeviceToHost);
    // assign_blocks<TS><<<32, 256, 0, stream1>>>(d_num_blocks, G->n, d_assignments);

    // cudaMemset(this->d_adj_labels, 0, sizeof(int) * G->n);

    // // count_lockfree5<NT, TS><<<nb, NT, 0, stream1>>>
    // //     (d_buffer1, d_labels, d_ptr, d_assignments, batch_n, g_table, d_tmp_labels, d_buffer2, batch_boundaries[j]);
    // count_lockfree5_2<NT, TS><<<nb, NT, 0, stream1>>>
    //     (this->d_neighbors, this->d_offsets, this->d_labels, d_assignments,
    //      G->n, this->d_tables, this->d_adj_labels, this->d_counter);


    int count = this->get_count();
    return count;
}


template<typename V, typename E>
void InCoreLFHT<V, E>::postprocess()
{
    this->free_gmem();
}


template<typename V, typename E>
void InCoreLFHT<V, E>::transfer_data()
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