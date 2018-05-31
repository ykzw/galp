/* -*- coding: utf-8 -*-
 *
 * nmi.h: Wed Jan  6 22:19:41 2016
 *
 */

#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <omp.h>


bool contain(const std::set<int> &s, int key)
{
    auto it = s.find(key);
    return it != s.end();
}


int get_intersection_size(const std::set<int> &lhs, const std::set<int> &rhs)
{
    int count = 0;

    if (lhs.size() < rhs.size()) {
        for (const auto v: lhs) {
            auto it = rhs.find(v);
            if (it != rhs.end()) {
                ++count;
            }
        }
    } else {
        for (const auto v: rhs) {
            auto it = lhs.find(v);
            if (it != lhs.end()) {
                ++count;
            }
        }
    }

    return count;
}


double compute_nmi
(const std::string gt_filename, const std::vector<int> &labels)
{
    std::vector<std::set<int> > gt_clusters;
    {  // Load ground truth clusters
        std::ifstream ifs(gt_filename);
        std::string s;
        int i = 0;
        while (getline(ifs, s)) {
            int v;
            std::stringstream ss(s);
            gt_clusters.push_back(std::set<int>());
            while (ss >> v) {
                gt_clusters[i].insert(v);
            }
            ++i;
        }
    }

    std::vector<std::set<int> > my_clusters;
    {  // Construct clusters from labels
        auto size = labels.size();

        std::set<int> label_set(labels.begin(), labels.end());
        auto num_labels = label_set.size();

        std::map<int, int> label_map;
        int i = 0;
        for (auto label: label_set) {
            auto it = label_map.find(label);
            if (it == label_map.end()) {
                label_map.insert(std::make_pair(label, i));
                ++i;
            }
        }

        my_clusters.resize(num_labels);
        for (int i = 0; i < size; ++i) {
            int mapped_label = label_map[labels[i]];
            my_clusters[mapped_label].insert(i);
        }
    }

    // Compute NMI
    int ca = gt_clusters.size();
    int cb = my_clusters.size();

    double Nt = 0.0;
    std::vector<double> Ni_(ca, 0.0);
    std::vector<double> N_j(cb, 0.0);
    double Ni_i = 0.0;
#pragma omp parallel reduction(+: Nt)
    {
        for (int i = 0; i < ca; ++i) {
#pragma omp barrier
#pragma omp single
            {
                Ni_i = 0;
            }
#pragma omp barrier

#pragma omp for reduction(+: Ni_i)
            for (int j = 0; j < cb; ++j) {
                int count = get_intersection_size(gt_clusters[i], my_clusters[j]);
                if (count > 0) {
                    double Nij = count;
                    Nt += Nij;
                    Ni_i += Nij;
                    N_j[j] += Nij;
                }
            }
#pragma omp barrier
#pragma omp single
            {
                Ni_[i] = Ni_i;
            }
        }
    }

    double v1 = 0.0;
#pragma omp parallel reduction(+: v1)
    {
        for (int i = 0; i < ca; ++i) {
#pragma omp for
            for (int j = 0; j < cb; ++j) {
                int count = get_intersection_size(gt_clusters[i], my_clusters[j]);
                if (count > 0) {
                    double Nij = count;
                    v1 += Nij * log2((Nij * Nt) / (Ni_[i] * N_j[j]));
                }
            }
        }
    }
    v1 *= -2;

    double v2 = 0.0;
    for (int i = 0; i < ca; ++i) {
        if (Ni_[i] > 0.0) {
            v2 += Ni_[i] * log2(Ni_[i] / Nt);
        }
    }
    for (int j = 0; j < cb; ++j) {
        if (N_j[j] > 0.0) {
            v2 += N_j[j] * log2(N_j[j] / Nt);
        }
    }

    printf("%d, %d\n", ca, cb);
    return v1 / v2;
}


double compute_f_measure
(const std::string gt_filename, const std::vector<int> &labels)
{
    std::vector<std::set<int> > gt_clusters;
    {  // Load ground truth clusters
        std::ifstream ifs(gt_filename);
        std::string s;
        int i = 0;
        while (getline(ifs, s)) {
            int v;
            std::stringstream ss(s);
            gt_clusters.push_back(std::set<int>());
            while (ss >> v) {
                gt_clusters[i].insert(v);
            }
            ++i;
        }
    }

    std::vector<std::set<int> > my_clusters;
    {  // Construct clusters from labels
        auto size = labels.size();

        std::set<int> label_set(labels.begin(), labels.end());
        auto num_labels = label_set.size();

        std::map<int, int> label_map;
        int i = 0;
        for (auto label: label_set) {
            auto it = label_map.find(label);
            if (it == label_map.end()) {
                label_map.insert(std::make_pair(label, i));
                ++i;
            }
        }

        my_clusters.resize(num_labels);
        for (int i = 0; i < size; ++i) {
            int mapped_label = label_map[labels[i]];
            my_clusters[mapped_label].insert(i);
        }
    }

    int ca = gt_clusters.size();
    int cb = my_clusters.size();
    int n = labels.size();

    double v = 0.0;
#pragma omp parallel for reduction(+: v)
    for (int i = 0; i < ca; ++i) {
        double max_f = 0.0;
        for (int j = 0; j < cb; ++j) {
            int count = get_intersection_size(gt_clusters[i], my_clusters[j]);
            double f = 2.0 * count / (gt_clusters[i].size() + my_clusters[j].size());
            max_f = std::max(max_f, f);
        }
        v += max_f * gt_clusters[i].size();
    }
    v /= n;

//     double v2 = 0.0;
// #pragma omp parallel for reduction(+: v2)
//     for (int j = 0; j < cb; ++j) {
//         double max_f = 0.0;
//         for (int i = 0; i < ca; ++i) {
//             int count = get_intersection_size(gt_clusters[i], my_clusters[j]);
//             double f = 2.0 * count / (gt_clusters[i].size() + my_clusters[j].size());
//             max_f = std::max(max_f, f);
//         }
//         v2 += max_f * my_clusters[j].size();
//     }
//     v2 /= n;
    // printf("%f, %f\n", v, v2);

    return v;
}


double compute_ari
(const std::string gt_filename, const std::vector<int> &labels)
{
    std::vector<std::set<int> > gt_clusters;
    {  // Load ground truth clusters
        std::ifstream ifs(gt_filename);
        std::string s;
        int i = 0;
        while (getline(ifs, s)) {
            int v;
            std::stringstream ss(s);
            gt_clusters.push_back(std::set<int>());
            while (ss >> v) {
                gt_clusters[i].insert(v);
            }
            ++i;
        }
    }

    std::vector<std::set<int> > my_clusters;
    {  // Construct clusters from labels
        auto size = labels.size();

        std::set<int> label_set(labels.begin(), labels.end());
        auto num_labels = label_set.size();

        std::map<int, int> label_map;
        int i = 0;
        for (auto label: label_set) {
            auto it = label_map.find(label);
            if (it == label_map.end()) {
                label_map.insert(std::make_pair(label, i));
                ++i;
            }
        }

        my_clusters.resize(num_labels);
        for (int i = 0; i < size; ++i) {
            int mapped_label = label_map[labels[i]];
            my_clusters[mapped_label].insert(i);
        }
    }

    int ca = gt_clusters.size();
    int cb = my_clusters.size();

    double v1 = 0.0;
#pragma omp parallel reduction(+: v1)
    for (int i = 0; i < ca; ++i) {
#pragma omp for
        for (int j = 0; j < cb; ++j) {
            double count = get_intersection_size(gt_clusters[i], my_clusters[j]);
            v1 += count * (count - 1.0) / 2.0;
        }
    }

    double v2 = 0.0;
#pragma omp parallel for reduction(+: v2)
    for (int i = 0; i < ca; ++i) {
        double s = 0.0;
        for (int j = 0; j < cb; ++j) {
            double count = get_intersection_size(gt_clusters[i], my_clusters[j]);
            s += count;
        }
        v2 += s * (s - 1.0) / 2.0;
    }

    double v3 = 0.0;
#pragma omp parallel for reduction(+: v3)
    for (int j = 0; j < cb; ++j) {
        double s = 0.0;
        for (int i = 0; i < ca; ++i) {
            double count = get_intersection_size(gt_clusters[i], my_clusters[j]);
            s += count;
        }
        v3 += s * (s - 1.0) / 2.0;
    }

    double n = labels.size();
    return (v1 - 2.0 * v2 * v3 / (n * (n - 1.0))) / ((v2 + v3) / 2.0 - 2.0 * v2 * v3 / (n * (n - 1.0)));
}
