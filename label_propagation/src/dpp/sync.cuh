// -*- coding: utf-8 -*-

#pragma once

#include "dpp.cuh"
#include "incore.cuh"
#include "../common/outofcore.cuh"


// GPU out-of-core without overlap, data-parallel primitives based label propagation
template<typename V, typename E>
class SyncDPP: public DPP<V, E>, public OutOfCore<V, E> {
public:
    using typename LabelPropagator<V, E>::GraphT;

    SyncDPP(std::shared_ptr<GraphT> _G, int bs)
        : DPP<V, E>(_G, (1 << bs), true), OutOfCore<V, E>(G, bs) { }
    virtual ~SyncDPP() = default;


private:
    // Methods
    void preprocess();
    int iterate(int i);
    void postprocess();

    // Attributes
    using LabelPropagator<V, E>::G;  // To avoid many "this->"es
};


template<typename V, typename E>
void SyncDPP<V, E>::preprocess()
{
    this->init_gmem(G->n, this->B);
    this->compute_batch_boundaries();
}


template<typename V, typename E>
int SyncDPP<V, E>::iterate(int i)
{
    int nbatches = this->get_num_batches();
    for (auto j: range(nbatches)) {
        int batch_n = this->get_num_batch_vertices(j);
        int batch_m = this->get_num_batch_edges(j);

        this->transfer_batch(j, this->d_neighbors, this->d_offsets);
        cudaDeviceSynchronize();

        this->perform_lp(batch_n, batch_m, this->bbs[j]);
        cudaDeviceSynchronize();
    }

    int count = this->get_count();
    return count;
}


template<typename V, typename E>
void SyncDPP<V, E>::postprocess()
{
    this->free_gmem();
}
