// -*- coding: utf-8 -*-

#pragma once

// Modified segmented reduce based on ModernGPU

template<typename T>
__device__ int binary_search(T key, T *data, int n)
{
    int begin = 0;
    int end = n;

    while (begin < end) {
        int mid = (begin + end) / 2;
        if (!(key < data[mid])) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }

    return begin;
}


__global__ void block_partition
(int nnz, int *ptr, int num_segments, int psize, int *partition, int num_partitions)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_partitions) {
        int p = min(tid * psize, nnz);

        int i;
        if (p == nnz) {
            i = num_segments;
        } else {
            i = binary_search(p, ptr, num_segments) - 1;
            if (p != ptr[i]) {
                i |= 0x80000000;
            }
        }

        partition[tid] = i;
    }
}



template<int TB, int ET>  // threads per block, elements per thread
__global__ void segmented_reduce
(const int *data, int n, int *ptr, int num_segments, int *partitions, int num_partitions,
 int *dst, int *satellite_dst, int *carry_outs, int *satellite_carry_outs)
{
    const int EB = TB * ET;
    const int tile_offset = EB * blockIdx.x;
    const int *tile_data = data + tile_offset;

    __shared__ int shared[EB + 1];
    __shared__ int shared2[EB + 1];

    int reg[ET];
    int reg2[ET];
    // global -> shared
#pragma unroll
    for (int i = 0; i < ET; ++i) {
        if (tile_offset + TB * i + threadIdx.x < n) {
            shared[TB * i + threadIdx.x] = tile_data[TB * i + threadIdx.x];
        } else {
            shared[TB * i + threadIdx.x] = 0;
        }
        shared2[TB * i + threadIdx.x] = tile_offset + TB * i + threadIdx.x;
    }
    __syncthreads();

    // shared -> register
#pragma unroll
    for (int i = 0; i < ET; ++i) {
        reg[i] = shared[ET * threadIdx.x + i];
        reg2[i] = tile_offset + ET * threadIdx.x + i;
    }
    __syncthreads();

    int part1 = partitions[blockIdx.x];
    int part2 = partitions[blockIdx.x + 1];
    int seg_begin = part1 & 0x7fffffff;
    int seg_end = part2 & 0x7fffffff;
    int seg_total = seg_end - seg_begin;
    bool flush = 0 == (0x80000000 & part2);
    seg_end += !flush;

    int nsegs = seg_end - seg_begin;
    int segs[ET + 1];
    int seg_starts[ET];
    int end_flags;
    int my_delta;
    if (seg_total > 0) {
        for (int i = threadIdx.x; i < nsegs; i += TB) {
            shared[i] = ptr[seg_begin + i];
        }
        __syncthreads();

        int end = flush ? (tile_offset + TB * ET) : INT_MAX;

        int my_offset = tile_offset + ET * threadIdx.x;
        int seg = binary_search(my_offset, shared, nsegs) - 1;

        int cur_offset = shared[seg];
        int next_offset = (seg + 1 < nsegs) ? shared[seg + 1] : end;

        segs[0] = seg;
        seg_starts[0] = cur_offset;

        end_flags = 0;
#pragma unroll
        for (int i = 1; i <= ET; ++i) {
            if (my_offset + i == next_offset) {
                end_flags |= 1 << (i - 1);
                ++seg;
                cur_offset = next_offset;
                next_offset = (seg + 1 < nsegs) ? shared[seg + 1] : end;
            }
            segs[i] = seg;
            if (i < ET) {
                seg_starts[i] = cur_offset;
            }
        }
        __syncthreads();


        const int nwarps = TB >> 5;
        const int warp_id = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const uint warp_mask = 0xffffffff >> (31 - lane);
        const uint block_mask = 0x7fffffff >> (31 - lane);

        uint warp_bits = __ballot_sync(0xffffffff, segs[0] != segs[ET]);
        shared[warp_id] = warp_bits;
        __syncthreads();

        if (threadIdx.x < nwarps) {
            uint block_bits = __ballot_sync(0xffffffff, shared[threadIdx.x] != 0);
            int warp_segment = 31 - __clz(block_mask & block_bits);
            int start = (warp_segment != -1) ?
                (31 - __clz(shared[warp_segment]) + 32 * warp_segment) : 0;
            shared[nwarps + threadIdx.x] = start;
        }
        __syncthreads();

        int start = 31 - __clz(warp_mask & warp_bits);
        if (start != -1) {
            start += threadIdx.x & ~31;
        } else {
            start = shared[nwarps + warp_id];
        }
        __syncthreads();

        my_delta = threadIdx.x - start;
    }

    if (seg_total > 0) {
        int s;
        int x;
        int my_scan[ET];
        int my_scan2[ET];

#pragma unroll
        for (int i = 0; i < ET; ++i) {
            // s = i ? s + reg[i] : reg[i];
            // s = i ? max(s, reg[i]) : reg[i];
            if (i > 0) {
                // if (s <= reg[i]) {
                if (s < reg[i]) {
                    s = reg[i];
                    x = reg2[i];
                }
            } else {
                s = reg[i];
                x = reg2[i];
            }
            my_scan[i] = s;
            my_scan2[i] = x;
            if (segs[i] != segs[i + 1]) {
                s = 0;  // Identity
                x = -1;
            }
        }

        int carry_out;
        int carry_out_s;
        int carry_in;
        int carry_in_s;
        {
            int first = 0;
            shared[first + threadIdx.x] = s;
            shared2[first + threadIdx.x] = x;
            __syncthreads();

#pragma unroll
            for (int offset = 1; offset < TB; offset += offset) {
                if (my_delta >= offset) {
                    // s = s + shared[first + threadIdx.x - offset];
                    // s = max(s, shared[first + threadIdx.x - offset]);
                    if (s <= shared[first + threadIdx.x - offset]) {
                    // if (s < shared[first + threadIdx.x - offset]) {
                        s = shared[first + threadIdx.x - offset];
                        x = shared2[first + threadIdx.x - offset];
                    }
                }
                first = TB - first;
                shared[first + threadIdx.x] = s;
                shared2[first + threadIdx.x] = x;
                __syncthreads();
            }

            //                                                  v Identity
            // s = threadIdx.x ? shared[first + threadIdx.x - 1] : 0;
            if (threadIdx.x > 0) {
                s = shared[first + threadIdx.x - 1];
                x = shared2[first + threadIdx.x - 1];
            } else {
                s = 0;
                x = -1;
            }
            carry_out = shared[first + TB - 1];
            carry_out_s = shared2[first + TB - 1];
            __syncthreads();
            carry_in = s;
            carry_in_s = x;
        }

        if (threadIdx.x == 0) {
            carry_outs[blockIdx.x] = carry_out;
            satellite_carry_outs[blockIdx.x] = carry_out_s;
        }

#pragma unroll
        for (int i = 0; i < ET; ++i) {
            // int s2 = carry_in + my_scan[i];
            // int s2 = max(carry_in, my_scan[i]);
            int s2;
            int x2;
            if (carry_in <= my_scan[i]) {
            // if (carry_in < my_scan[i]) {
                s2 = my_scan[i];
                x2 = my_scan2[i];
            } else {
                s2 = carry_in;
                x2 = carry_in_s;
            }
            if (segs[i] != segs[i + 1]) {
                shared[segs[i]] = s2;
                shared2[segs[i]] = x2;
                carry_in = 0;  // Identity
                carry_in_s = -1;
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < seg_total; i += TB) {
            dst[seg_begin + i] = shared[i];
            satellite_dst[seg_begin + i] = shared2[i];
        }
    } else {
        int s;
        int x;
#pragma unroll
        for (int i = 0; i < ET; ++i) {
            // s = i ? s + reg[i] : reg[i];
            // s = i ? max(s, reg[i]) : reg[i];
            if (i > 0) {
                if (s <= reg[i]) {
                // if (s < reg[i]) {
                    s = reg[i];
                    x = reg2[i];
                }
            } else {
                s = reg[i];
                x = reg2[i];
            }
        }
        shared[threadIdx.x] = s;
        shared2[threadIdx.x] = x;
        __syncthreads();

#pragma unroll
        for (int i = TB / 2; i >= 1; i >>= 1) {
            if (threadIdx.x < i) {
                // s = s + shared[threadIdx.x + i];
                // s = max(s, shared[threadIdx.x + i]);
                if (s <= shared[threadIdx.x + i]) {
                // if (s < shared[threadIdx.x + i]) {
                    s = shared[threadIdx.x + i];
                    x = shared2[threadIdx.x + i];
                }
                shared[threadIdx.x] = s;
                shared2[threadIdx.x] = x;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            carry_outs[blockIdx.x] = shared[0];
            satellite_carry_outs[blockIdx.x] = shared2[0];
        }
    }
}


template<int TB>
__global__ void segmented_reduce_spine
(const int *partitions, int n, int *dst, int *satellite_dst,
 const int *carry_ins, const int *satellite_carry_ins, int *carry_outs, int *satellite_carry_outs)
{
    __shared__ int shared[TB * 2];
    __shared__ int shared2[TB * 2];

    const int gid = TB * blockIdx.x + threadIdx.x;
    int seg1 = (gid < n) ? (partitions[gid] & 0x7fffffff) : INT_MAX;
    int seg2 = (gid + 1 < n) ? (partitions[gid + 1] & 0x7fffffff) : INT_MAX;

    int carry_in = (gid < n) ? carry_ins[gid] : 0;
    int carry_in_s = (gid < n) ? satellite_carry_ins[gid] : -1;
    int d = (gid < n) ? dst[seg1] : 0;
    int d2 = (gid < n) ? satellite_dst[seg1] : -1;

    bool end_flag = seg1 != seg2;

    int carry_out;
    int carry_out_s;
    int my_delta;
    {
        const int nwarps = TB >> 5;
        const int warp_id = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const uint warp_mask = 0xffffffff >> (31 - lane);
        const uint block_mask = 0x7fffffff >> (31 - lane);

        uint warp_bits = __ballot_sync(0xffffffff, end_flag);
        shared[warp_id] = warp_bits;
        __syncthreads();

        if (threadIdx.x < nwarps) {
            uint block_bits = __ballot_sync(0xffffffff, shared[threadIdx.x] != 0);
            int warp_segment = 31 - __clz(block_mask & block_bits);
            int start = (warp_segment != -1) ?
                (31 - __clz(shared[warp_segment]) + 32 * warp_segment) : 0;
            shared[nwarps + threadIdx.x] = start;
        }
        __syncthreads();

        int start = 31 - __clz(warp_mask & warp_bits);
        if (start != -1) {
            start += threadIdx.x & ~31;
        } else {
            start = shared[nwarps + warp_id];
        }
        __syncthreads();

        my_delta = threadIdx.x - start;
    }

    int s, x;
    {
        int first = 0;
        s = carry_in;
        x = carry_in_s;
        shared[first + threadIdx.x] = s;
        shared2[first + threadIdx.x] = x;
        __syncthreads();

#pragma unroll
        for (int offset = 1; offset < TB; offset += offset) {
            if (my_delta >= offset) {
                // s = s + shared[first + threadIdx.x - offset];
                // s = max(s, shared[first + threadIdx.x - offset]);
                if (s <= shared[first + threadIdx.x - offset]) {
                // if (s < shared[first + threadIdx.x - offset]) {
                    s = shared[first + threadIdx.x - offset];
                    x = shared2[first + threadIdx.x - offset];
                }
            }
            first = TB - first;
            shared[first + threadIdx.x] = s;
            shared2[first + threadIdx.x] = x;
            __syncthreads();
        }

        s = threadIdx.x ? shared[first + threadIdx.x - 1] : 0;
        x = threadIdx.x ? shared2[first + threadIdx.x - 1] : -1;
        carry_out = shared[first + TB - 1];
        carry_out_s = shared2[first + TB - 1];
        __syncthreads();
    }

    if (end_flag) {
        // dst[seg1] = s + d;
        // dst[seg1] = max(s, d);
        if (s <= d) {
        // if (s < d) {
            dst[seg1] = d;
            satellite_dst[seg1] = d2;
        } else {
            dst[seg1] = s;
            satellite_dst[seg1] = x;
        }
    }

    if (threadIdx.x == 0) {
        carry_outs[blockIdx.x] = carry_out;
        satellite_carry_outs[blockIdx.x] = carry_out_s;
    }
}



template<int TB>
__global__ void segmented_reduce_spine2
(const int *partitions, int num_blocks, int n, int eb, int *dst, int *satellite_dst,
 const int *carry_ins, const int *satellite_carry_ins)
{
    __shared__ int shared[TB * 2];
    __shared__ int shared2[TB * 2];
    __shared__ int s_carry_in;
    __shared__ int s_carry_in_s;
    __shared__ int s_carry_in_seg;

    for (int i = 0; i < num_blocks; i += TB) {
        int gid = (i + threadIdx.x) * eb;

        int seg1 = (gid < n) ? (0x7fffffff & partitions[gid]) : INT_MAX;
        int seg2 = (gid + eb < n) ? (0x7fffffff & partitions[gid + eb]) : INT_MAX;
        int carry_in = (i + threadIdx.x < num_blocks) ? carry_ins[i + threadIdx.x]: 0;
        int carry_in_s = (i + threadIdx.x < num_blocks) ? satellite_carry_ins[i + threadIdx.x]: 0;
        int d = (gid < n) ? dst[seg1] : 0;
        int d_s = (gid < n) ? satellite_dst[seg1] : -1;

        bool end_flag = seg1 != seg2;
        int my_delta;
        {
            const int nwarps = TB >> 5;
            const int warp_id = threadIdx.x >> 5;
            const int lane = threadIdx.x & 31;
            const uint warp_mask = 0xffffffff >> (31 - lane);
            const uint block_mask = 0x7fffffff >> (31 - lane);

            uint warp_bits = __ballot_sync(0xffffffff, end_flag);
            shared[warp_id] = warp_bits;
            __syncthreads();

            if (threadIdx.x < nwarps) {
                uint block_bits = __ballot_sync(0xffffffff, shared[threadIdx.x] != 0);
                int warp_segment = 31 - __clz(block_mask & block_bits);
                int start = (warp_segment != -1) ?
                    (31 - __clz(shared[warp_segment]) + 32 * warp_segment) : 0;
                shared[nwarps + threadIdx.x] = start;
            }
            __syncthreads();

            int start = 31 - __clz(warp_mask & warp_bits);
            if (start != -1) {
                start += threadIdx.x & ~31;
            } else {
                start = shared[nwarps + warp_id];
            }
            __syncthreads();

            my_delta = threadIdx.x - start;
        }

        int carry_out;
        int carry_out_s;
        int s, x;
        {
            int first = 0;
            s = carry_in;
            x = carry_in_s;
            shared[first + threadIdx.x] = s;
            shared2[first + threadIdx.x] = x;
            __syncthreads();

#pragma unroll
            for (int offset = 1; offset < TB; offset += offset) {
                if (my_delta >= offset) {
                    // s = s + shared[first + threadIdx.x - offset];
                    // s = max(s, shared[first + threadIdx.x - offset]);
                    if (s <= shared[first + threadIdx.x - offset]) {
                    // if (s < shared[first + threadIdx.x - offset]) {
                        s = shared[first + threadIdx.x - offset];
                        x = shared2[first + threadIdx.x - offset];
                    }
                }
                first = TB - first;
                shared[first + threadIdx.x] = s;
                shared2[first + threadIdx.x] = x;
                __syncthreads();
            }

            s = threadIdx.x ? shared[first + threadIdx.x - 1] : 0;
            x = threadIdx.x ? shared2[first + threadIdx.x - 1] : -1;
            carry_out = shared[first + TB - 1];
            carry_out_s = shared2[first + TB - 1];
            __syncthreads();
        }

        if (end_flag) {
            if (i > 0 && seg1 == s_carry_in_seg) {
                // s = s + shared[TB * 2 + 1];
                // s = max(s, shared[TB * 2 + 1]);
                if (s <= s_carry_in) {
                // if (s < s_carry_in) {
                    s = s_carry_in;
                    x = s_carry_in_s;
                }
            }
            // dst[seg1] = s + d;
            // dst[seg1] = max(s, d);
            if (s <= d) {
            // if (s < d) {
                dst[seg1] = d;
                satellite_dst[seg1] = d_s;
            } else {
                dst[seg1] = s;
                satellite_dst[seg1] = x;
            }
        }

        if (i + TB < num_blocks) {
            __syncthreads();
            if (i > 0) {
                if (TB - 1 == threadIdx.x) {
                    // shared[TB * 2 + 1] = (shared[TB * 2] == seg2) ?
                    //     shared[TB * 2 + 1] + carry_out : carry_out;
                    // shared[TB * 2 + 1] = (shared[TB * 2] == seg2) ?
                    //     max(shared[TB * 2 + 1], carry_out) : carry_out;
                    if (s_carry_in_seg == seg2) {
                        if (s_carry_in <= carry_out) {
                        // if (s_carry_in < carry_out) {
                            s_carry_in = carry_out;
                            s_carry_in_s = carry_out_s;
                        }
                    } else {
                        s_carry_in = carry_out;
                        s_carry_in_s = carry_out_s;
                    }
                    s_carry_in_seg = seg2;
                }
            } else {
                if (TB - 1 == threadIdx.x) {
                    s_carry_in = carry_out;
                    s_carry_in_s = carry_out_s;
                    s_carry_in_seg = seg2;
                }
            }
            __syncthreads();
        }
    }
}


struct SegmentedReducer {
    static constexpr int TB = 256;      // Threads per block
    static constexpr int ET = 17;       // Elements per thread
    static constexpr int EB = TB * ET;  // Elements per block

    SegmentedReducer() { }
    SegmentedReducer(int N, cudaStream_t s=0): stream(s) {
        int num_tiles = (N + EB - 1) / EB;
        int num_tiles2 = (num_tiles + TB - 1) / TB;

        cudaMalloc((void **) &partitions, sizeof(int) * (num_tiles + 1));

        cudaMalloc((void **) &carry_outs, sizeof(int) * num_tiles);
        cudaMalloc((void **) &satellite_carry_outs, sizeof(int) * num_tiles);

        cudaMalloc((void **) &carry_outs2, sizeof(int) * num_tiles2);
        cudaMalloc((void **) &satellite_carry_outs2, sizeof(int) * num_tiles2);
    }

    ~SegmentedReducer() {
        if (partitions) cudaFree(partitions);
        if (carry_outs) cudaFree(carry_outs);
        if (satellite_carry_outs) cudaFree(satellite_carry_outs);
        if (carry_outs2) cudaFree(carry_outs2);
        if (satellite_carry_outs2) cudaFree(satellite_carry_outs2);
    }

    void apply(int *data, int N, int *ptr, int M, int *dst, int *satellite_dst) {
        {
            const int num_tiles = (N + EB - 1) / EB + 1;
            const int num_blocks = (num_tiles + 64 - 1) / 64;
            block_partition<<<num_blocks, 64, 0, stream>>>
                (N, ptr, M, EB, partitions, num_tiles);
        }

        {
            const int num_blocks = (N + EB - 1) / EB;
            segmented_reduce<TB, ET><<<num_blocks, TB, 0, stream>>>
                (data, N, ptr, M, partitions, num_blocks,
                 dst, satellite_dst, carry_outs, satellite_carry_outs);
        }

        {
            const int num_tiles = (N + EB - 1) / EB;
            const int num_blocks = (num_tiles + TB - 1) / TB;

            segmented_reduce_spine<TB><<<num_blocks, TB, 0, stream>>>
                (partitions, num_tiles, dst, satellite_dst, carry_outs, satellite_carry_outs,
                 carry_outs2, satellite_carry_outs2);

            if (num_blocks > 1) {
                segmented_reduce_spine2<TB><<<1, TB, 0, stream>>>
                    (partitions, num_blocks, num_tiles, TB, dst, satellite_dst,
                     carry_outs2, satellite_carry_outs2);
            }
        }
    }

    int *partitions;

    int *carry_outs;
    int *satellite_carry_outs;

    int *carry_outs2;
    int *satellite_carry_outs2;

    cudaStream_t stream;
};
