// -*- coding: utf-8 -*-

#pragma once

#include "graph.h"
#include "outofcore.cuh"


template<typename V, typename E, typename S>
class SyncLP: public S, public OutOfCore<V, E> {
public:
    using typename OutOfCore<V, E>::GraphT;

    SyncLP(std::shared_ptr<GraphT> _G, int bs)
        : S(_G, bs), OutOfCore<V, E>(_G, bs) { }
    SyncLP(std::shared_ptr<GraphT> _G, int p, int bs)
        : S(_G, p), OutOfCore<V, E>(_G, bs) { }
    virtual ~SyncLP() = default;


private:
    // Methods
    void preprocess();
    int iterate(int i);
    void postprocess();

    // Attributes
    using LabelPropagator<V, E>::G;

};


template<typename V, typename E, typename S>
void SyncLP<V, E, S>::preprocess()
{
    this->init_gmem(G->n, this->B);
    this->compute_batch_boundaries();
}


template<typename V, typename E, typename S>
int SyncLP<V, E, S>::iterate(int i)
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


template<typename V, typename E, typename S>
void SyncLP<V, E, S>::postprocess()
{
    this->free_gmem();
}
