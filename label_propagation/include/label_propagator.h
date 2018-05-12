// -*- coding: utf-8 -*-

#pragma once

#include <memory>
#include <random>
#include <cstdlib>
#include "graph.h"


template<typename V, typename E>
class LabelPropagator {
public:
    using GraphT = CSRGraph<V, E>;

    LabelPropagator() = default;
    LabelPropagator(std::shared_ptr<GraphT> _G)
        : G(_G), labels(new V[_G->n]) { }

    virtual ~LabelPropagator() = default;

    virtual std::pair<double, double> run(int niter);

    std::vector<V> get_labels() {
        auto p = labels.get();
        std::vector<V> tmp_labels(p, p + G->n);
        return tmp_labels;
    }

// protected:
    std::shared_ptr<GraphT> G;
    std::unique_ptr<V[]> labels;

};


template<typename V, typename E>
std::pair<double, double> LabelPropagator<V, E>::run(int niter)
{
    Timer t; t.start();

    std::vector<V> vertices(G->n);

    for (auto i: range(G->n)) {
        labels[i] = i;
        vertices[i] = i;
    }

    for (auto i: range(niter)) {
        std::random_shuffle(vertices.begin(), vertices.end());

        V nupdates = 0;
        for (auto u: vertices) {
            std::map<V, int> label_count;
            V max_label = labels[u];
            int max_count = 0;

            for (auto v: G->iterate_neighbors(u)) {
                V label = labels[v];
                int c = ++(label_count[label]);
                if ((max_count < c) ||
                    (max_count == c && ((double) rand() / RAND_MAX) < 0.5)) {
                    max_count = c;
                    max_label = label;
                }
            }

            if (labels[u] != max_label) {
                labels[u] = max_label;
                ++nupdates;
            }
        }
        if (nupdates <= G->n * 1e-5) {
            break;
        }
    }

    t.stop();

    auto f = t.elapsed_time();
    return {f, f};
}
