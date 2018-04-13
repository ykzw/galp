// -*- coding: utf-8 -*-

#pragma once

#include <deque>
#include <algorithm>
#include "omp.h"

#include "dpp.cuh"
#include "incore.cuh"
#include "../common/outofcore.cuh"
#include "hybrid_cpu_worker.cuh"


using Task = int;
using TaskQueue = std::deque<Task>;
using TaskQueueList = std::vector<TaskQueue>;


// CPU--GPU hybrid, GPU out-of-core with overlap, data-parallel primitives based label propagation
template<typename V, typename E>
class HybridDPP: public DPP<V, E>, public OutOfCore<V, E> {
public:
    using typename LabelPropagator<V, E>::GraphT;

    HybridDPP(std::shared_ptr<GraphT> _G, int bs)
        : DPP<V, E>(_G, (1 << bs), true), OutOfCore<V, E>(G, bs), cpu_worker(G, this->bbs, tql, qlock, this->labels.get()) { }
    virtual ~HybridDPP() = default;

    std::pair<double, double> run(int niter);


private:
    // Methods
    void preprocess();
    int iterate(int i);
    void postprocess();

    void swap_buffers();

    int gpu_work(int i);
    std::pair<Task, Task> get_new_GPU_tasks(int i);
    void make_task_queue_list(int niter);

    void init_gmem(int n, int B);
    void free_gmem();

    // Attributes
    using LabelPropagator<V, E>::G;  // To avoid many "this->"es

    // For double buffering
    int *d_neighbors_buf;  // B
    int *d_offsets_buf;    // n + 1

    cudaStream_t stream1;
    cudaStream_t stream2;

    TaskQueueList tql;
    omp_lock_t qlock;

    CPUWorker<V, E> cpu_worker;
};


template<typename V, typename E>
std::pair<double, double> HybridDPP<V, E>::run(int niter)
{
    this->compute_batch_boundaries();
    make_task_queue_list(niter);

    return DPP<V, E>::run(niter);
}


template<typename V, typename E>
void HybridDPP<V, E>::preprocess()
{
    this->init_gmem(G->n, this->B);
    cpu_worker.d_labels = this->d_labels;

    stream1 = this->context->Stream();
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

    omp_init_lock(&qlock);

    cudaHostRegister((void *) this->labels.get(), sizeof(V) * G->n, cudaHostRegisterMapped);
}


template<typename V, typename E>
int HybridDPP<V, E>::iterate(int i)
{
    int count = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        // The last thread handles the GPU
        bool gpu_thread = tid == nthreads - 1;

        int c;
        if (gpu_thread) { // GPU
            printf("Iteration %d\n", i + 1);
            c = gpu_work(i);
        } else if (tid < nthreads - 1) {  // CPU
            c = cpu_worker.run(i, tid);
        }
        if (tid == 0 || gpu_thread) {
            #pragma omp atomic
            count += c;
        }
    }

    return count;
}


template<typename V, typename E>
void HybridDPP<V, E>::postprocess()
{
    free_gmem();

    omp_destroy_lock(&qlock);

    cudaHostUnregister((void *) this->labels.get());

    // cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}


template<typename V, typename E>
int HybridDPP<V, E>::gpu_work(int i)
{
    Timer gput; gput.start();
    int npb = 0;

    if (i == 0) {
        auto t = tql[0].front();
        this->transfer_batch(t, this->d_neighbors, this->d_offsets, stream1);
        cudaDeviceSynchronize();
    }

    while (true) {
        auto tasks = get_new_GPU_tasks(i);
        int task = tasks.first;
        int next = tasks.second;
        if (task == -1) {
            break;
        }
        ++npb;

        int batch_n = this->get_num_batch_vertices(task);
        int batch_m = this->get_num_batch_edges(task);

        if (next >= 0) {
            // The next batch
            this->transfer_batch(next, d_neighbors_buf, d_offsets_buf, stream2);
        }

        this->perform_lp(batch_n, batch_m, this->bbs[task], stream1, &this->h_norm_offsets[G->n + 1]);
        cudaStreamSynchronize(stream2);

        swap_buffers();

        // Update the this->labels on the CPU main memory
        cudaMemcpyAsync(this->labels.get() + this->bbs[task], this->d_labels + this->bbs[task],
                        sizeof(int) * (this->bbs[task + 1] - this->bbs[task]),
                        cudaMemcpyDeviceToHost, stream2);
    }

    gput.stop();
    printf("GPU processed %d batches in %fs\n", npb, gput.elapsed_time());

    int count = this->get_count();
    return count;
}


template<typename V, typename E>
std::pair<Task, Task> HybridDPP<V, E>::get_new_GPU_tasks(int i)
{
    int cur, next;

    omp_set_lock(&qlock);
    if (!tql[i].empty()) {
        cur = tql[i].front();
        tql[i].pop_front();
        if (!tql[i].empty()) {
            next = tql[i].front();
        } else {
            next = tql[i + 1].front();
        }
    } else {
        cur = -1;
    }
    omp_unset_lock(&qlock);

    return {cur, next};
}


template<typename V, typename E>
void HybridDPP<V, E>::make_task_queue_list(int niter)
{
    tql.clear();
    tql.resize(niter + 1);

    TaskQueue tmp_queue;
    int nbatches = this->get_num_batches();
    for (auto j: range(nbatches)) {
        tmp_queue.push_back(j);
    }

    for (auto i: range(niter)) {
        std::random_shuffle(tmp_queue.begin(), tmp_queue.end());

        for (auto j: range(nbatches)) {
            tql[i].push_back(tmp_queue[j]);
        }
    }

    tql[niter].push_back(-1);
}


template<typename V, typename E>
void HybridDPP<V, E>::swap_buffers()
{
    std::swap(this->d_neighbors, d_neighbors_buf);
    this->d_adj_labels = this->d_neighbors;
    std::swap(this->d_offsets, d_offsets_buf);
}


template<typename V, typename E>
void HybridDPP<V, E>::init_gmem(int n, int B)
{
    cudaMalloc(&d_neighbors_buf, sizeof(int) * B);
    cudaMalloc(&d_offsets_buf, sizeof(int) * (n + 1));

    DPP<V, E>::init_gmem(n, B);
}


template<typename V, typename E>
void HybridDPP<V, E>::free_gmem()
{
    cudaFree(d_neighbors_buf);
    cudaFree(d_offsets_buf);

    DPP<V, E>::free_gmem();
}