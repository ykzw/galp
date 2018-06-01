// -*- coding: utf-8 -*-

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <set>
#include <algorithm>


void convert_edgelist_txt(const std::string &filename)
{
    std::vector<std::vector<int>> adjacent_list;
    std::set<int> vertex_set;

    std::string s;

    long progress = 0;
    std::ifstream ifs(filename);
    while (getline(ifs, s)) {
        if (s[0] == '#') {
            continue;
        }
        int u, v;
        sscanf(s.data(), "%d\t%d\n", &u, &v);
        vertex_set.insert(u);
        vertex_set.insert(v);
        if (u >= adjacent_list.size() || v >= adjacent_list.size()) {
            adjacent_list.resize(std::max(u, v) + 1);
        }
        adjacent_list[u].push_back(v);
        adjacent_list[v].push_back(u);

        if (((++progress) & 0xffffff) == 0) {
            printf("%ld\n", progress);
        }
    }

    std::vector<int> vertex_list(vertex_set.begin(), vertex_set.end());
    std::sort(vertex_list.begin(), vertex_list.end());
    std::map<int, int> vertex_mapping;
    for (int i = 0; i < vertex_list.size(); ++i) {
        vertex_mapping[vertex_list[i]] = i;
    }

    std::ofstream ofs("graph.bin", std::ofstream::out | std::ofstream::binary);
    int writebuf[2];
    for (int i = 0; i < vertex_list.size(); ++i) {
        if ((i & 0xffffff) == 0) {
            printf("%d / %d\n", i + 1, vertex_list.size());
        }
        auto u = vertex_list[i];
        std::sort(adjacent_list[u].begin(), adjacent_list[u].end());
        auto x = vertex_mapping[u];
        writebuf[0] = x;
        for (int j = 0; j < adjacent_list[u].size(); ++j) {
            auto y = vertex_mapping[adjacent_list[u][j]];
            if (x >= y) continue;
            writebuf[1] = y;
            ofs.write((char *) writebuf, sizeof(int) * 2);
        }
    }
}


void convert_edgelist_txt_simple(const std::string &filename)
{
    std::string s;
    std::ifstream ifs(filename);
    std::vector<std::vector<int>> adjacency_list;
    while (getline(ifs, s)) {
        int u, v;
        sscanf(s.data(), "%d %d\n", &u, &v);

        if (adjacency_list.size() <= v) {
            adjacency_list.resize(v + 1, std::vector<int>());
        }

        adjacency_list[u].push_back(v);
        adjacency_list[v].push_back(u);
    }

    std::ofstream ofs("graph.bin", std::ofstream::out | std::ofstream::binary);
    int writebuf[2];
    int n = adjacency_list.size();
    for (int i = 0; i < n; ++i) {
        int m = adjacency_list[i].size();
        writebuf[0] = i;
        for (int j = 0; j < m; ++j) {
            writebuf[1] = adjacency_list[i][j];
            ofs.write((char *) writebuf, sizeof(int) * 2);
        }
    }
}


int main(int argc, char *argv[])
{

    if (argc != 2) {
        printf("usage: ./edgelist2bin filename\n");
        return 0;
    }

    const std::string filename(argv[1]);

    // convert_edgelist_txt_simple(filename);
    convert_edgelist_txt(filename);


    return 0;
}
