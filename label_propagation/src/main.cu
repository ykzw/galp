// -*- coding: utf-8 -*-

#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cstdio>
#include <cstdint>

#include "myutil.h"
#include "graph.h"

#include "dpp/incore.cuh"
#include "dpp/sync.cuh"
#include "dpp/async.cuh"
#include "dpp/incore_li.cuh"
#include "dpp/hybrid.cuh"

#include "lfht/incore.cuh"
#include "lfht/async.cuh"

#include "../../graph_clustering_orig/nmi.h"


template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


int main(int argc, char *argv[])
{
    if (argc < 5) {
        printf("usage: ./label_propagation {0: async, 1: sync, 2: li, 3: hybrid} graph_file iterations buffersize [ground_truth_file]\n");
        return 1;
    }

    const int niter = atoi(argv[3]);
    const int buffer_size = atoi(argv[4]);

    const int mode = atoi(argv[1]);
    const char *graph_file = argv[2];

    using Vertex = int32_t;
    using Edge = int64_t;
    using GraphT = CSRGraph<Vertex, Edge>;
    using Propagator = LabelPropagator<Vertex, Edge>;

    // Load a graph
    std::shared_ptr<GraphT> graph(new CSRGraph<Vertex, Edge>(graph_file));
    std::unique_ptr<Propagator> propagator;

    const Vertex n = graph->n;
    const Edge m = graph->m;
    std::cout << "Data is ready on the host (" << n << ", " << m << ")" << std::endl;

    std::string proc_name;

    if (mode != 2 && mode < 4 && m < (1 << buffer_size)) {
        proc_name = "dpp incore";
        propagator = make_unique<InCoreDPP<Vertex, Edge>>(graph);
    } else {
        switch (mode) {
        case 0:
            proc_name = "dpp async";
            propagator = make_unique<AsyncDPP<Vertex, Edge>>(graph, buffer_size);
            break;
        case 1:
            proc_name = "dpp sync";
            propagator = make_unique<SyncDPP<Vertex, Edge>>(graph, buffer_size);
            break;
        case 2:
            proc_name = "dpp li";
            propagator = make_unique<InCoreLIDPP<Vertex, Edge>>(graph);
            break;
        case 3:
            proc_name = "dpp hybrid";
            propagator = make_unique<HybridDPP<Vertex, Edge>>(graph, buffer_size);
            break;
        case 4:
            proc_name = "lfht async";
            propagator = make_unique<AsyncLFHT<Vertex, Edge>>(graph, buffer_size);
            // propagator = make_unique<InCoreLFHT<Vertex, Edge>>(graph);
            break;
        }
    }
    std::pair<double, double> result = propagator->run(niter);

    // for (auto i: range(100)) {
    //     printf("%d, %d\n", i, propagator->labels[i]);
    // }

    double f1 = result.first;
    double f2 = result.second;

    if (argc == 6) {  // Check the accuracy
        double nmi = 0.0;
        double fm = 0.0;
        double ari = 0.0;

        auto labels = propagator->get_labels();
        const char *ground_truth_file = argv[5];

        Timer tnmi; tnmi.start();
        nmi = compute_nmi(ground_truth_file, labels);
        tnmi.stop();

        Timer tfm; tfm.start();
        fm = compute_f_measure(ground_truth_file, labels);
        tfm.stop();

        Timer tari; tari.start();
        ari = compute_ari(ground_truth_file, labels);
        tari.stop();

        fprintf(stderr, "%s\t%s\t%d\t%d\t%f\t%f\t%f\t%f\t%f\n",
                proc_name.c_str(), argv[2], niter, buffer_size, f1, f2, nmi, fm, ari);
    } else {
        fprintf(stderr, "%s\t%s\t%d\t%d\t%f\t%f\n", proc_name.c_str(), argv[2], niter, buffer_size, f1, f2);
    }

    return 0;
}
