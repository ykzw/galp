#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def main():
    adjacent_dict = {}

    graph_file = sys.argv[1]

    print("Data loading")
    lines = 0
    with open(graph_file) as f:
        for lines, line in enumerate(f, 1):
            pass

    vertex_set = set()
    with open(graph_file) as f:
        for i, line in tqdm(iterable=enumerate(f), total=lines, ascii=True, leave=True):
            u, v = [int(v) for v in line.strip().split()]
            vertex_set.add(u)
            vertex_set.add(v)
            if u in adjacent_dict:
                adjacent_dict[u].add(v)
            else:
                adjacent_dict[u] = set([v])
            if v in adjacent_dict:
                adjacent_dict[v].add(u)
            else:
                adjacent_dict[v] = set([u])

    vertex_list = sorted(vertex_set)
    vertex_mapping = {}
    for i, u in enumerate(vertex_list):
        vertex_mapping[u] = i

    print("Converting")
    with open("normalized_" + graph_file, "w") as f:
        for i, u in tqdm(enumerate(vertex_list), total=len(vertex_list), ascii=True, leave=True):
            if u not in adjacent_dict:
                continue
            for v in sorted(adjacent_dict[u]):
                if vertex_mapping[u] >= vertex_mapping[v]:
                    continue
                f.write("{0} {1}\n".format(vertex_mapping[u], vertex_mapping[v]))


    try:
        ground_truth_file = sys.argv[2]
    except IndexError:
        return

    print("Loading ground truth data")
    clusters = {}
    with open(ground_truth_file) as rf:
        for line in tqdm(rf, total=len(vertex_set), ascii=True, leave=True):
            u, v = line.strip().split()
            u, v = vertex_mapping[int(u)], vertex_mapping[int(v)]
            if v in clusters:
                clusters[v].add(u)
            else:
                clusters[v] = set([u])

    print("Writing converted ground truth data")
    with open("normalized_" + ground_truth_file, "w") as wf:
        for clst in tqdm(clusters.itervalues(), total=len(clusters), ascii=True, leave=True):
            wf.write(" ".join(str(v) for v in sorted(clst)) + "\n")


if __name__ == '__main__':
    main()
