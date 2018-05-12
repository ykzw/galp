// -*- coding: utf-8 -*-

#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cstdio>
#include <cstdint>
#include <unistd.h>

#include "myutil.h"
#include "graph.h"

#include "dpp/variants.cuh"
#include "dpp/incore_li.cuh"

#include "lfht/variants.cuh"

#include "../nmi.h"


void usage(char *prog)
{
    std::cerr << "Usage: " << prog << " [options] method policy data [test]" << std::endl
              << "  method: Specify the method to be used." << std::endl
              << "    0 -- data-parallel primitives" << std::endl
              << "    1 -- lock-free hash tables" << std::endl
              << "  policy: Specify the policy to perform the method." << std::endl
              << "    0 -- GPU in-core" << std::endl
              << "    1 -- Out-of-core without overlap" << std::endl
              << "    2 -- Out-of-core with overlap" << std::endl
              << "    3 -- CPU-GPU hybrid" << std::endl
              << "    4 -- Depending on the method" << std::endl
              << "      method=0: Load-imbalanced" << std::endl
              << "      method=1: Multi-GPU, out-of-core with overlap" << std::endl
              << "    5 -- Multi-GPU in-core (only for method 1)" << std::endl
              << "  data: The graph data file." << std::endl
              << "  test: The ground-truth file. Accuracy is computed if supplied." << std::endl
              << "Options: " << std::endl
              << " -b n: Set buffer sizes to 2^n. The default is 24." << std::endl
              << " -i n: Specify the number of iterations. The default is 10." << std::endl
              << " -g n: Use n GPUs. The default is 1." << std::endl
        ;
    exit(1);
}


template<typename V, typename E>
void check_gpu_mem(V n, E m, int ngpus)
{
    // Check whether the GPU memory is enough.
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (free < sizeof(int) * (3 * m / ngpus + 2 * n)) {
        std::cout << "Not enough GPU memory!!" << std::endl
                  << "  GPU memory: " << free / 1024.0 / 1024 / 1024 << " GB" << std::endl
                  << "    Required: " << sizeof(int) * (3.0 * m / ngpus + 2 * n) / 1024 / 1024 / 1024 << " GB" << std::endl;
        exit(1);
    }
}


template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


int main(int argc, char *argv[])
{
    int ch;
    extern char	*optarg;
    extern int optind, opterr;

    int buffer_pow = 24;
    int niter = 10;
    int ngpus = 1;
    int lfht_policy = 1;
    while ((ch = getopt(argc, argv, "b:i:g:l:")) != -1) {
        switch (ch) {
        case 'b':
            buffer_pow = atoi(optarg);
            break;

        case 'i':
            niter = atoi(optarg);
            break;

        case 'g':
            ngpus = atoi(optarg);
            break;

        case 'l':
            lfht_policy = atoi(optarg);
            break;

        default:
            usage(argv[0]);

        }
    }

    argc -= optind;
    if (argc < 3) {
        usage(argv[0]);
    }
    argv += optind;

    int method = atoi(argv[0]);
    int policy = atoi(argv[1]);
    char *graph_file = argv[2];

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

    const Vertex n = graph->n;
    const Edge m = graph->m;
    std::cout << "Data is ready on the host (" << n << ", " << m << ")" << std::endl;

    std::string proc_name;
    std::unique_ptr<Propagator> propagator;
    switch (method * 5 + policy) {
    case 0:
        buffer_pow = 0;
        proc_name = "dpp\tincore";
        check_gpu_mem(n, m, 1);
        propagator = make_unique<InCoreDPP<Vertex, Edge>>(graph);
        break;

    case 1:
        proc_name = "dpp\tsync";
        propagator = make_unique<SyncDPP<Vertex, Edge>>(graph, buffer_pow);
        break;

    case 2:
        proc_name = "dpp\tasync";
        propagator = make_unique<AsyncDPP<Vertex, Edge>>(graph, buffer_pow);
        break;

    case 3:
        proc_name = "dpp\thybrid";
        propagator = make_unique<HybridDPP<Vertex, Edge>>(graph, buffer_pow);
        break;

    case 4:
        buffer_pow = 0;
        proc_name = "dpp\tli";
        check_gpu_mem(n, m, 1);
        propagator = make_unique<InCoreLIDPP<Vertex, Edge>>(graph);
        break;

    case 5:
        buffer_pow = 0;
        proc_name = "lfht" + std::to_string(lfht_policy) + "\tincore";
        check_gpu_mem(n, m, 1);
        propagator = make_unique<InCoreLFHT<Vertex, Edge>>(graph, lfht_policy);
        break;

    case 6:
        proc_name = "lfht" + std::to_string(lfht_policy) + "\tsync";
        propagator = make_unique<SyncLFHT<Vertex, Edge>>(graph, lfht_policy, buffer_pow);
        break;

    case 7:
        proc_name = "lfht" + std::to_string(lfht_policy) + "\tasync";
        propagator = make_unique<AsyncLFHT<Vertex, Edge>>(graph, lfht_policy, buffer_pow);
        break;

    case 8:
        proc_name = "lfht" + std::to_string(lfht_policy) + "\thybrid";
        propagator = make_unique<HybridLFHT<Vertex, Edge>>(graph, lfht_policy, buffer_pow);
        break;

    case 9:
        proc_name = "lfht" + std::to_string(lfht_policy) + "\tasync multi" + std::to_string(ngpus);
        propagator = make_unique<MultiAsyncLFHT<Vertex, Edge>>(graph, ngpus, lfht_policy, buffer_pow);
        break;

    case 10:
        buffer_pow = 0;
        proc_name = "lfht" + std::to_string(lfht_policy) + "\tincore multi" + std::to_string(ngpus);
        check_gpu_mem(n, m, ngpus);
        propagator = make_unique<MultiInCoreLP<Vertex, Edge>>(graph, ngpus, lfht_policy);
        break;

    default:
        buffer_pow = 0;
        proc_name = "serial\t ";
        propagator = make_unique<LabelPropagator<Vertex, Edge>>(graph);
    }
    std::cout << "----------" << proc_name << "----------" << std::endl;
    std::pair<double, double> result = propagator->run(niter);

    // for (auto i: range(300)) {
    //     printf("%d, %d\n", i, propagator->labels[i]);
    // }

    double f1 = result.first;
    double f2 = result.second;

    double nmi = 0.0;
    double fm = 0.0;
    double ari = 0.0;
    if (argc == 4) {  // Check the accuracy
        auto labels = propagator->get_labels();
        const char *ground_truth_file = argv[3];

        Timer tnmi; tnmi.start();
        nmi = compute_nmi(ground_truth_file, labels);
        tnmi.stop();

        Timer tfm; tfm.start();
        fm = compute_f_measure(ground_truth_file, labels);
        tfm.stop();

        Timer tari; tari.start();
        ari = compute_ari(ground_truth_file, labels);
        tari.stop();
    }

    const char *filename = basename(argv[2]);
    fprintf(stderr, "%s\t%s\t%d\t%d\t%f\t%f\t%f\t%f\t%f\n",
            proc_name.c_str(), filename, niter, buffer_pow, f1, f2, nmi, fm, ari);


    return 0;
}
