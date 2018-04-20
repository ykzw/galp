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

#include "dpp/variants.cuh"
#include "dpp/incore_li.cuh"

#include "lfht/variants.cuh"

#include "../nmi.h"


template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}



int main(int argc, char *argv[])
{
    if (argc < 7) {
        printf("usage: ./label_propagation {0: async, 1: sync, 2: li, 3: hybrid} graph_file iterations buffersize [ground_truth_file]\n");
        return 1;
    }

    const int mode = atoi(argv[1]);
    const int lfht_policy = atoi(argv[2]);
    const int ngpus = atoi(argv[3]);
    const char *graph_file = argv[4];

    const int niter = atoi(argv[5]);
    const int buffer_size = atoi(argv[6]);

    using Vertex = int32_t;
    using Edge = int64_t;
    using GraphT = CSRGraph<Vertex, Edge>;
    using Propagator = LabelPropagator<Vertex, Edge>;

    // Warm up
    #pragma omp parallel for num_threads(ngpus)
    for (int i = 0; i < ngpus; ++i) {
        cudaSetDevice(i);
        cudaFree(0);
    }

    // Load a graph
    std::shared_ptr<GraphT> graph(new CSRGraph<Vertex, Edge>(graph_file));
    std::unique_ptr<Propagator> propagator;

    const Vertex n = graph->n;
    const Edge m = graph->m;
    std::cout << "Data is ready on the host (" << n << ", " << m << ")" << std::endl;

    std::string proc_name;

    if (m < (1 << buffer_size)) {
        // Data is smaller than the buffer
        if (mode < 4) {
            proc_name = "dpp incore";
            propagator = make_unique<InCoreDPP<Vertex, Edge>>(graph);
        } else {
            proc_name = std::string("lfht incore ") + std::string(argv[2]);
            propagator = make_unique<InCoreLFHT<Vertex, Edge>>(graph, lfht_policy);
        }
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
            proc_name = "dpp hybrid";
            propagator = make_unique<HybridDPP<Vertex, Edge>>(graph, buffer_size);
            break;
        case 3:
            proc_name = "dpp li";
            propagator = make_unique<InCoreLIDPP<Vertex, Edge>>(graph);
            break;
        case 4:
            proc_name = std::string("lfht async ") + std::string(argv[2]);
            propagator = make_unique<AsyncLFHT<Vertex, Edge>>(graph, lfht_policy, buffer_size);
            break;
        case 5:
            proc_name = std::string("lfht sync ") + std::string(argv[2]);
            propagator = make_unique<SyncLFHT<Vertex, Edge>>(graph, lfht_policy, buffer_size);
            break;
        case 6:
            proc_name = std::string("lfht hybrid ") + std::string(argv[2]);
            propagator = make_unique<HybridLFHT<Vertex, Edge>>(graph, lfht_policy, buffer_size);
            break;
        case 7:
            proc_name = std::string("lfht multi async ") + std::string(argv[2]);
            propagator = make_unique<MultiAsyncLFHT<Vertex, Edge>>(graph, ngpus, lfht_policy, buffer_size);
            break;
        case 8:
            proc_name = std::string("lfht multi incore ") + std::string(argv[2]);
            // propagator = make_unique<MultiInCoreLFHT<Vertex, Edge>>(graph, ngpus, lfht_policy);
            propagator = make_unique<MultiInCoreLP<Vertex, Edge>>(graph, ngpus, lfht_policy);
            break;
        }
    }
    std::pair<double, double> result = propagator->run(niter);

    for (auto i: range(100)) {
        printf("%d, %d\n", i, propagator->labels[i]);
    }

    double f1 = result.first;
    double f2 = result.second;

    const char *filename = basename(argv[4]);
    if (argc > 7) {  // Check the accuracy
        double nmi = 0.0;
        double fm = 0.0;
        double ari = 0.0;

        auto labels = propagator->get_labels();
        const char *ground_truth_file = argv[7];

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
                proc_name.c_str(), filename, niter, buffer_size, f1, f2, nmi, fm, ari);
    } else {
        fprintf(stderr, "%s\t%s\t%d\t%d\t%f\t%f\n",
                proc_name.c_str(), filename, niter, buffer_size, f1, f2);
    }

    return 0;
}
