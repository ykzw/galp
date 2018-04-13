// -*- coding: utf-8 -*-

#pragma once

#include "dpp.cuh"


// GPU in-core, data-parallel primitives based label propagation
template<typename V, typename E>
class InCoreDPP: public DPP<V, E> {
public:
    using typename LabelPropagator<V, E>::GraphT;

    InCoreDPP(std::shared_ptr<GraphT> _G): DPP<V, E>(_G) { }
    virtual ~InCoreDPP() = default;


protected:
    // Methods
    void preprocess();
    int iterate(int i);
    void postprocess();

    void transfer_data();

    // Attributes
    using LabelPropagator<V, E>::G;  // To avoid many "this->"es

};


template<typename V, typename E>
void InCoreDPP<V, E>::preprocess()
{
    this->init_gmem(G->n, G->m);
    transfer_data();
}


template<typename V, typename E>
int InCoreDPP<V, E>::iterate(int i)
{
    this->perform_lp(G->n, G->m);

    int count = this->get_count();
    return count;
}


template<typename V, typename E>
void InCoreDPP<V, E>::postprocess()
{
    this->free_gmem();
}


template<typename V, typename E>
void InCoreDPP<V, E>::transfer_data()
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