// -*- coding: utf-8 -*-

#pragma once


template<typename IntType>
class _Range {
public:
    __host__ __device__ constexpr _Range(IntType e): _start(0), _end(e), _step(1) { }
    __host__ __device__ constexpr _Range(IntType s, IntType e): _start(std::min(s, e)), _end(e), _step(1) { }
    // It's invalid to pass arguments such that
    // IntType is unsigned and e < s (i.e., e - s becomes negative)
    __host__ __device__ constexpr _Range(IntType s, IntType e, IntType p):
        _start(s), _end((e - s + p - 1) / p > 0 ? ((e - s) / p + ((e - s) % p != 0)) * p + s : s), _step(p) { }

    class iterator: public std::iterator<std::input_iterator_tag, IntType> {
    public:
        __host__ __device__ constexpr iterator(IntType s, const _Range &parent): cur(s), parent(parent) { }
        __host__ __device__ constexpr IntType operator*() const { return cur; }
        __host__ __device__ iterator &operator++() { cur += parent._step; return *this; };
        __host__ __device__ const iterator operator++(int) {
            cur += parent._step;
            return iterator(cur - parent._step, parent);
        }

        __host__ __device__ constexpr bool operator==(const iterator &rhs) const { return cur == rhs.cur; }
        __host__ __device__ constexpr bool operator!=(const iterator &rhs) const { return cur != rhs.cur; }

    private:
        IntType cur;
        const _Range &parent;
    };

    __host__ __device__ constexpr iterator begin() const { return iterator(_start, *this); }
    __host__ __device__ constexpr iterator end() const { return iterator(_end, *this); }

private:
    const IntType _start;
    const IntType _end;
    const IntType _step;
};



// GPU-specific functions >>>>>>>>>>>>>

template<typename IntType>
__device__ constexpr inline _Range<IntType> grid_stride_range(IntType e)
{
    return _Range<IntType>(blockDim.x * blockIdx.x + threadIdx.x, e, gridDim.x * blockDim.x);
}

template<typename IntType>
__device__ constexpr inline _Range<IntType> block_stride_range(IntType e)
{
    return _Range<IntType>(threadIdx.x, e, blockDim.x);
}

template<typename IntType>
__device__ constexpr inline _Range<IntType> block_stride_range(IntType b, IntType e)
{
    return _Range<IntType>(b + threadIdx.x, e, blockDim.x);
}


template<typename IntType, typename Function>
__device__ inline void apply_range(_Range<IntType> r, Function f)
{
    for (const auto i: r) {
        f(i);
    }
}



template<int S, int E, typename Function>
__device__ inline void apply_thread_unroll(Function f)
{
#pragma unroll
    for (int i = S; i < E; ++i) {
        f(i);
    }
}

template<int E, typename Function>
__device__ inline void apply_thread_unroll(Function f)
{
    apply_thread_unroll<0, E>(f);
}


template<int E, typename Function>
__device__ inline void apply_thread_unroll_bound(Function f, int e)
{
#pragma unroll
    for (int i = 0; i < E; ++i) {
        int index = blockDim.x * i + threadIdx.x;
        if (index >= e) break;
        f(i);
    }
}



template<int S, int E, typename Function>
__device__ inline void apply_block_unroll(Function f)
{
#pragma unroll
    for (int i = S; i < E; ++i) {
        int index = blockDim.x * i + threadIdx.x;
        f(index);
    }
}

template<int E, typename Function>
__device__ inline void apply_block_unroll(Function f)
{
    apply_block_unroll<0, E>(f);
}


template<typename Function>
__device__ inline void apply_block_bound(Function f, int e)
{
    for (int i = threadIdx.x; i < e; i += blockDim.x) {
        f(i);
    }
}

template<int E, typename Function>
__device__ inline void apply_block_unroll_bound(Function f, int e)
{
#pragma unroll
    for (int i = 0; i < E; ++i) {
        int index = blockDim.x * i + threadIdx.x;
        if (index >= e) break;
        f(index);
    }
}
