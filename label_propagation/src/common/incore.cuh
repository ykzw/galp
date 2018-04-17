// -*- coding: utf-8 -*-

#pragma once

#include "graph.h"
#include "label_propagator.h"


// GPU in-core, lock-free hash table based label propagation
template<typename V, typename E, typename S>
class InCoreLP: public S {
public:
    using typename LabelPropagator<V, E>::GraphT;

    InCoreLP(std::shared_ptr<GraphT> _G): S(_G, _G->m, false) { }
    InCoreLP(std::shared_ptr<GraphT> _G, int p): S(_G, p, false) { }
    virtual ~InCoreLP() = default;


protected:
    // Methods
    void preprocess();
    int iterate(int i);
    void postprocess();

    void transfer_data();

    // Attributes
    using LabelPropagator<V, E>::G;

};


template<typename V, typename E, typename S>
void InCoreLP<V, E, S>::preprocess()
{
    this->init_gmem(G->n, G->m);
    transfer_data();
}


template<typename V, typename E, typename S>
int InCoreLP<V, E, S>::iterate(int i)
{
    this->perform_lp(G->n, G->m);

    int count = this->get_count();
    return count;
}


template<typename V, typename E, typename S>
void InCoreLP<V, E, S>::postprocess()
{
    this->free_gmem();
}


template<typename V, typename E, typename S>
void InCoreLP<V, E, S>::transfer_data()
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