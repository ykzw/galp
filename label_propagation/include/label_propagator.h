// -*- coding: utf-8 -*-

#pragma once

#include <memory>
#include <random>
#include "graph.h"


template<typename V, typename E>
class LabelPropagator {
public:
    using GraphT = CSRGraph<V, E>;

    LabelPropagator() = default;
    LabelPropagator(std::shared_ptr<GraphT> _G)
        : G(_G), labels(new V[_G->n]) { }

    virtual ~LabelPropagator() = default;

    virtual std::pair<double, double> run(int K) = 0;

    std::vector<V> get_labels() {
        auto p = labels.get();
        std::vector<V> tmp_labels(p, p + G->n);
        return tmp_labels;
    }

// protected:
    std::shared_ptr<GraphT> G;
    std::unique_ptr<V[]> labels;

};
