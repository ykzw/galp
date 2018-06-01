// -*- coding: utf-8 -*-

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <set>
#include <algorithm>


void convert_metis_txt(const std::string &filename)
{
    std::vector<int> vertex_id;

    std::string s;

    long progress = 0;
    std::ifstream ifs(filename);
    getline(ifs, s);
    while (getline(ifs, s)) {
        std::stringstream ss(s);
        int v;
        while (ss >> v) {
            if (vertex_id.size() <= v) {
                vertex_id.resize(v + 1, 0);
            }
            vertex_id[v] = 1;
        }
        ++progress;
        if ((progress & 0xfffff) == 0) {
            std::cout << progress << std::endl;
        }
    }

    printf("A\n");

    int a = 0;
    for (int i = 0; i < vertex_id.size(); ++i) {
        int b = vertex_id[i];
        vertex_id[i] = a;
        a += b;
    }

    printf("B\n");

    std::vector<std::vector<int>> adjacency_list(a);
    ifs.clear();
    ifs.seekg(0, std::ios_base::beg);
    long i = 1;
    getline(ifs, s);
    while (getline(ifs, s)) {
        std::stringstream ss(s);
        int u = vertex_id[i];
        long j;
        while (ss >> j) {
            int v = vertex_id[j];
            adjacency_list[u].push_back(v);
            // adjacency_list[v].push_back(u);
        }
        ++i;
        if ((i & 0xfffff) == 0) {
            std::cout << i << std::endl;
        }
    }

    printf("C\n");

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
        if ((i & 0xfffff) == 0) {
            std::cout << i << std::endl;
        }
    }
}


int main(int argc, char *argv[])
{

    if (argc != 2) {
        printf("usage: ./metis2bin filename\n");
        return 0;
    }

    const std::string filename(argv[1]);

    convert_metis_txt(filename);


    return 0;
}
