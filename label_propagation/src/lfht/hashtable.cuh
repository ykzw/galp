// -*- coding: utf-8 -*-

#pragma once

#include "../common/range.cuh"


__device__ uint32_t hash(uint32_t k) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
}


__device__ uint32_t _increment
(uint32_t *keys, uint32_t *vals, uint32_t begin, uint32_t end, uint32_t key, uint32_t x=1)
{
    auto s = end - begin;
    for (uint32_t i = hash(key) % s;; ++i) {
        if (i == s) i = 0;

        uint32_t k = keys[begin + i];
        if (k != key) {
            if (k != 0) {
                continue;
            }

            uint32_t prev = atomicCAS(&keys[begin + i], 0, key);
            if ((prev != 0) && (prev != key)) {
                continue;
            }
        }

        auto v = atomicAdd(&vals[begin + i], x);
        return v + x;
    }
}


class GlobalHT {
public:
    uint32_t *keys;
    uint32_t *vals;

    __device__ uint32_t increment(uint32_t begin, uint32_t end, uint32_t key, uint32_t x=1) {
        return _increment(keys, vals, begin, end, key, x);
    }
};


template<int C>
class SharedHT {
public:
    uint32_t keys[C];
    uint32_t vals[C];
    int nitems;

    __device__ uint32_t increment(uint32_t key, uint32_t x=1) {
        auto c = _increment(keys, vals, 0, C, key, x);
        if (c - x == 0) {
            atomicAdd(&nitems, 1);
        }
        return c;
    }

    __device__ void clear() {
        if (threadIdx.x == 0) {
            nitems = 0;
        }

        for (auto i: block_stride_range(C)) {
            keys[i] = 0;
            vals[i] = 0;
        }
    }

};