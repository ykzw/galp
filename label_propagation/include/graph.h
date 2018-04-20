// -*- coding: utf-8 -*-

#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include "myutil.h"



template<typename V, typename E>
class Graph {
    // Virtual class representing a graph
    // * V is the type of vertices
    // * E is the type of edges

public:
    struct iterator {
        iterator(V *first, V *last): _begin(first), _end(last) { }
        iterator(V *first, std::ptrdiff_t size): iterator(first, first + size) {}

        constexpr V *begin() const noexcept { return _begin; }
        constexpr V *end() const noexcept { return _end; }

        V *_begin;
        V *_end;
    };

    Graph() = default;
    Graph(const std::string &filename);
    virtual ~Graph() = default;

    virtual Range<V> iterate_vertices() const = 0;
    virtual iterator iterate_neighbors(V v) = 0;

    V n;  // Number of vertices
    E m;  // Number of undirected edges
};


template<typename V>
using AdjacencyList = std::vector<std::vector<V>>;


// Returns a graph as an adjacency list
template<typename V>
AdjacencyList<V> load_edgelist_txt(const std::string &filename);
template<typename V>
AdjacencyList<V> load_metisgraph_txt(const std::string &filename);


template<typename V, typename E>
class CSRGraph: public Graph<V, E> {
    // Class representing a graph by the CSR format

public:
    using iterator = typename Graph<V, E>::iterator;

    CSRGraph() = default;
    CSRGraph(const std::string &filename) {
        auto pos = filename.find_last_of(".");
        auto ext = filename.substr(pos);
        if (ext == ".txt" || ext == ".dat") {
            auto adj_list = load_edgelist_txt<V>(filename);

            this->n = adj_list.size();
            auto size_accumulator = [](const E &a, decltype(adj_list[0]) &b) {
                return a + b.size();
            };
            // this->m = std::accumulate(adj_list.begin(), adj_list.end(),
            //                           E(), size_accumulator) + this->n;
            // //                                                 ^ loops
            this->m = std::accumulate(adj_list.begin(), adj_list.end(), E(), size_accumulator);

            neighbors.resize(this->m);
            offsets.resize(this->n + 1);

            // Adjacency list to CSR format
            E cur = 0;
            offsets[0] = 0;
            for (auto u: range(this->n)) {
                bool f = false;
                for (auto v: adj_list[u]) {
                    if (f && u < v) {
                        // // Insert a loop
                        // neighbors[cur++] = u;
                        f = false;
                    }
                    neighbors[cur++] = v;
                }
                if (f) {
                    neighbors[cur++] = u;
                }
                offsets[u + 1] = cur;
                adj_list[u].clear();
            }
        } else {
            load_symmetric_edgelist_bin(filename);
            this->n = offsets.size() - 1;
            this->m = neighbors.size();
        }

        cudaHostRegister((void *) &neighbors[0], sizeof(V) * this->m,
                         cudaHostRegisterMapped | cudaHostRegisterPortable);
        cudaHostRegister((void *) &offsets[0], sizeof(E) * (this->n + 1),
                         cudaHostRegisterMapped | cudaHostRegisterPortable);
    }

    ~CSRGraph() {
        cudaHostUnregister((void *) &neighbors[0]);
        cudaHostUnregister((void *) &offsets[0]);
    }

    void load_symmetric_edgelist_bin(const std::string &filename) {
        std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
        auto length = ifs.tellg() / sizeof(int);
        ifs.seekg(0, std::ios::beg);
        const int bufsize = 2;
        V buf[bufsize];

        int cur = -1;
        bool f = false;
        E i;
        for (i = 0; i < length; i += bufsize) {
            if ((i & 0xffffffelu) == 0) {
                printf("%ld / %ld (%.2f%%)\n", i + 1, length, 100.0 * (i + 1.0) / length);
            }
            ifs.read((char *) buf, sizeof(int) * bufsize);
            V u = buf[0];
            V v = buf[1];
            if (u != cur) {
                offsets.push_back(neighbors.size());
                cur = u;
                f = true;
            }
            if (f && u < v) {
                // // Insert a loop
                // neighbors.push_back(u);
                f = false;
            }
            neighbors.push_back(v);
        }

        offsets.push_back(neighbors.size());
    }

    Range<V> iterate_vertices() const {
        return range(0, this->n);
    }

    iterator iterate_neighbors(V v) {
        return iterator(&neighbors[offsets[v]], &neighbors[offsets[v + 1]]);
    }

    std::vector<V> neighbors;
    std::vector<E> offsets;

};


template<typename V>
AdjacencyList<V> load_edgelist_txt(const std::string &filename)
{
    std::ifstream ifs(filename);
    AdjacencyList<V> adj_list;

    // Read the file line by line
    std::string s;
    while (getline(ifs, s)) {
        V u, v;
        std::stringstream ss(s);
        ss >> u >> v;
        if (adj_list.size() <= v) {
            adj_list.resize(v + 1);
        }
        adj_list[u].push_back(v);
        // Symmetrize
        // * Assume the edge (u, v), where u > v, is not in the file
        adj_list[v].push_back(u);
    }

    return adj_list;
}
