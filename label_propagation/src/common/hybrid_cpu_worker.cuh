#pragma once

#include <deque>
#include <map>
#include <unordered_map>
#include <unistd.h>


// 15418.courses.cs.cmu.edu/spring2013/article/43
struct Barrier {
    int counter;
    int flag;
    int nthreads;
    std::vector<std::vector<int>> sense;
    omp_lock_t lock;


    Barrier(int n): counter(0), flag(0), nthreads(n), sense(n, {0}) {
        omp_init_lock(&lock);
    }

    ~Barrier() { omp_destroy_lock(&lock); }

    void sync(int tid) {
        sense[tid][0] = 1 - sense[tid][0];
        omp_set_lock(&lock);
        int arrived = ++counter;
        if (arrived == nthreads) {
            omp_unset_lock(&lock);
            counter = 0;
            flag = sense[tid][0];
        } else {
            omp_unset_lock(&lock);
            while (flag != sense[tid][0]) {
                usleep(1);
            }
        }
    }
};


template<typename V, typename E>
struct CPUWorker {
    using Task = int;
    using TaskQueue = std::deque<Task>;
    using TaskQueueList = std::vector<TaskQueue>;

    CPUWorker
    (std::shared_ptr<CSRGraph<V, E>> _G, const std::vector<int> &_bbs,
     TaskQueueList &_tql, omp_lock_t &_qlock, V *_h_labels)
        : G(_G.get()), bbs(_bbs), tql(_tql), qlock(_qlock), h_labels(_h_labels) {
        #pragma omp parallel
        {
            #pragma omp single
            nworkers = omp_get_num_threads() - 1;

            #pragma omp for
            for (int i = 0; i < G->n; ++i) {
                // h_labels[i] = i;
                h_labels[i] = G->neighbors[(G->offsets[i] + G->offsets[i + 1]) / 2];
            }
        }
        barrier = new Barrier(nworkers);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~CPUWorker() {
        delete barrier;
        cudaStreamDestroy(stream);
    }


    int run(int i, int tid);
    void set_new_task(int i);
    bool update_label(V v, E begin, E end);


    // Data shared with GPU workers
    CSRGraph<V, E> *G;
    const std::vector<int> &bbs;  // batch boundaries
    TaskQueueList &tql;  // task queue list
    omp_lock_t &qlock;  // task queue list
    V *h_labels;
    int *d_labels;

    // Data specific to CPU workers and shared by them
    int nworkers;
    int task;
    int npb;
    Barrier *barrier;
    int nupdated = 0;
    cudaStream_t stream;
};



template<typename V, typename E>
int CPUWorker<V, E>::run(int i, int tid)
{
    Timer cput;
    if (tid == 0) {
        cput.start();
        npb = 0;
        nupdated = 0;
    }

    int my_nupdated = 0;
    while (true) {
        if (tid == 0) {
            set_new_task(i);
        }
        #pragma omp flush(task)

        barrier->sync(tid);

        if (task == -1) {
            break;
        }

        V vstart = bbs[task];
        V vend = bbs[task + 1];

        #pragma omp for schedule(dynamic, 1) nowait
        for (V v = vstart; v < vend; ++v) {
            bool updated = update_label(v, G->offsets[v], G->offsets[v + 1]);
            if (updated) {
                ++my_nupdated;
            }
        }

        barrier->sync(tid);

        if (tid == 0) {
            cudaMemcpyAsync(d_labels + bbs[task], h_labels + bbs[task],
                            sizeof(int) * (bbs[task + 1] - bbs[task]), cudaMemcpyHostToDevice, stream);
            // cudaDeviceSynchronize();
        }
    }

    if (tid == 0) {
        cput.stop();
        printf("CPU processed %d batches in %fs\n", npb, cput.elapsed_time());
    }

    if (my_nupdated > 0) {
        #pragma omp atomic
        nupdated += my_nupdated;
    }

    return nupdated;
}


template<typename V, typename E>
void CPUWorker<V, E>::set_new_task(int i)
{
    omp_set_lock(&qlock);
    if (tql[i].size() > 3) {
        // Keep several batches for the GPU worker
        // Because if the CPU takes the last batch, the entire process can become slow
        // (GPU is much faster)
        task = tql[i].back();
        tql[i].pop_back();
        ++npb;
    } else {
        task = -1;
    }
    omp_unset_lock(&qlock);
}


template<typename V, typename E>
bool CPUWorker<V, E>::update_label(V v, E begin, E end)
{
    std::map<V, int> label_count;
    V max_label = h_labels[v];
    int max_count = 0;

    for (E k = begin; k < end; ++k) {
        V u = G->neighbors[k];
        V label = h_labels[u];
        int c = ++(label_count[label]);
        if (max_count <= c) {
            max_count = c;
            max_label = label;
        }
    }

    if (h_labels[v] != max_label) {
        h_labels[v] = max_label;
        return true;
    } else {
        return false;
    }
}
