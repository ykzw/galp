// -*- coding: utf-8 -*-

#pragma once

#include "lfht.cuh"
// #include "incore.cuh"
#include "../common/outofcore.cuh"


// GPU out-of-core with overlap, data-parallel primitives based label propagation
template<typename V, typename E>
class AsyncLFHT: public LFHTBase<V, E>, public OutOfCore<V, E> {
public:
    using typename LabelPropagator<V, E>::GraphT;

    AsyncLFHT(std::shared_ptr<GraphT> _G, int bs)
        : LFHTBase<V, E>(_G, true), OutOfCore<V, E>(_G, bs) { }
    virtual ~AsyncLFHT() = default;


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

    int *d_num_blocks;     // n + 1
    int2 *d_assignments;   // B
    int *d_max_counts;     // B + 1

    // For double buffering
    int *d_neighbors_buf;  // B
    int *d_offsets_buf;    // n + 1

    cudaStream_t stream1;
    cudaStream_t stream2;

};


template<typename V, typename E>
void AsyncLFHT<V, E>::preprocess()
{
    init_gmem(G->n, this->B);

    this->compute_batch_boundaries();

    stream1 = this->context->Stream();
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
}


template<typename V, typename E>
int AsyncLFHT<V, E>::iterate(int i)
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

        cudaMemsetAsync(this->d_tables.keys, 0, sizeof(uint32_t) * batch_m, stream1);
        cudaMemsetAsync(this->d_tables.vals, 0, sizeof(uint32_t) * batch_m, stream1);
        // cudaMemsetAsync(d_max_counts, 0, sizeof(int) * batch_n, stream1);

        // this->perform_lp(batch_n, batch_m, this->bbs[j], stream1, &this->h_norm_offsets[G->n + 1]);
        // count_lockfree2_2<64><<<batch_n, 64, 0, stream1>>>(this->d_neighbors, this->d_offsets, this->d_labels,
        //                                                    this->d_tables, this->d_counter, this->bbs[j]);
        update_lockfree_smem<32, 32 * 15><<<batch_n, 32, 0, stream1>>>
            (this->d_neighbors, this->d_offsets, this->d_labels, this->d_tables, this->d_counter, this->bbs[j]);

        // const int NT = 64;
        // const int TS = NT * 3;
        // compute_num_blocks<TS><<<divup(batch_n, 256), 256, 0, stream1>>>(this->d_offsets, batch_n, d_num_blocks);
        // mgpu::ScanExc(d_num_blocks, batch_n + 1, *this->context);
        // int nb;
        // cudaMemcpyAsync(&this->h_norm_offsets[G->n + 1], d_num_blocks + batch_n, sizeof(int), cudaMemcpyDeviceToHost, stream1);
        // cudaStreamSynchronize(stream1);
        // nb = this->h_norm_offsets[G->n + 1];
        // assign_blocks<TS><<<32, 256, 0, stream1>>>(d_num_blocks, batch_n, d_assignments);

        // // count_lockfree5<NT, TS><<<nb, NT, 0, stream1>>>
        // //     (d_buffer1, d_labels, d_ptr, d_assignments, batch_n, g_table, d_tmp_labels, d_buffer2, batch_boundaries[j]);
        // update_lockfree_smem_lb<NT, TS><<<nb, NT, 0, stream1>>>
        //     (this->d_neighbors, this->d_offsets, this->d_labels,
        //      d_assignments, this->d_tables, d_max_counts, this->d_counter, this->bbs[j]);

        cudaDeviceSynchronize();

        swap_buffers();
    }

    int count = this->get_count();
    return count;
}


template<typename V, typename E>
void AsyncLFHT<V, E>::postprocess()
{
    free_gmem();

    // cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}


template<typename V, typename E>
void AsyncLFHT<V, E>::swap_buffers()
{
    std::swap(this->d_neighbors, d_neighbors_buf);
    this->d_adj_labels = this->d_neighbors;
    std::swap(this->d_offsets, d_offsets_buf);
}


template<typename V, typename E>
void AsyncLFHT<V, E>::init_gmem(int n, int B)
{
    cudaMalloc(&d_neighbors_buf, sizeof(int) * B);
    cudaMalloc(&d_offsets_buf, sizeof(int) * (n + 1));

    cudaMalloc(&d_num_blocks, sizeof(int) * (n + 1));
    cudaMalloc(&d_assignments, sizeof(int2) * B);
    cudaMalloc(&d_max_counts, sizeof(int) * B);

    LFHTBase<V, E>::init_gmem(n, B);
}


template<typename V, typename E>
void AsyncLFHT<V, E>::free_gmem()
{
    cudaFree(d_neighbors_buf);
    cudaFree(d_offsets_buf);

    LFHTBase<V, E>::free_gmem();
}