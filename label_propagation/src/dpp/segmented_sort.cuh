// -*- coding: utf-8 -*-

using namespace mgpu;

struct SegmentedSorter {
    typedef LaunchBoxVT<
        128, 11, 0,
        128, 11, 0,
        128, (sizeof(int) > 4) ? 7 : 11, 0
    > Tuning;

    SegmentedSorter(int size, ContextPtr c): context(*c) {
        int2 launch = Tuning::GetLaunchParams(context);
        const int NV = launch.x * launch.y;

        mem = AllocSegSortBuffers(size, NV, support, true, context);
        destDevice = context.Malloc<int>(size);

        partitionsDevice = context.Malloc<int>(MGPU_DIV_UP(size, NV) + 1);
    }

    void run(int *data_global, int count, const int* indices_global, int indicesCount);
    void clear(int n, int nv);
    template<MgpuBounds Bounds, typename It1, typename Comp>
    void partition(int, It1, int, int, Comp, CudaContext &);

    CudaContext &context;
    SegSortSupport support;
    MGPU_MEM(byte) mem;
    MGPU_MEM(int) destDevice;
    MGPU_MEM(int) partitionsDevice;
};


void SegmentedSorter::run
(int *data_global, int count, const int* indices_global, int indicesCount)
{
    auto comp = less<int>();
    bool verbose = false;

    const bool Stable = true;
    int2 launch = Tuning::GetLaunchParams(context);
    const int NV = launch.x * launch.y;

    int numBlocks = MGPU_DIV_UP(count, NV);
    int numPasses = FindLog2(numBlocks, true);

    clear(count, NV);

    int* source = data_global;
    int* dest = destDevice->get();

    partition<MgpuBoundsLower>(count, indices_global, indicesCount, NV, mgpu::less<int>(), context);

    KernelSegBlocksortIndices<Tuning, Stable, false>
        <<<numBlocks, launch.x, 0, context.Stream()>>>
        (source, (int *) nullptr, count, indices_global, partitionsDevice->get(),
         (1 & numPasses) ? dest : source, (int *) nullptr, support.ranges_global, comp);

    if (1 & numPasses) std::swap(source, dest);

    SegSortPasses<Tuning, true, false>
        (support, source, (int *) nullptr, count, numBlocks, numPasses, dest, (int *) nullptr, comp, context, verbose);
}


void SegmentedSorter::clear(int count, int nv)
{
    int numBlocks = MGPU_DIV_UP(count, nv);
    int numPasses = FindLog2(numBlocks, true);
    int numRanges = 1;
    int numBlocks2 = MGPU_DIV_UP(numBlocks, 2);
    for (int pass = 1; pass < numPasses; ++pass) {
        numRanges += numBlocks2;
        numBlocks2 = MGPU_DIV_UP(numBlocks2, 2);
    }

    int rangesSize = MGPU_ROUND_UP_POW2(sizeof(int) * numBlocks, 128);
    int ranges2Size = MGPU_ROUND_UP_POW2(sizeof(int2) * numRanges, 128);
    int mergeListSize = MGPU_ROUND_UP_POW2(sizeof(int4) * numBlocks, 128);
    int copyListSize = MGPU_ROUND_UP_POW2(sizeof(int) * numBlocks, 128);
    int countersSize = MGPU_ROUND_UP_POW2(sizeof(int4), 128);
    int copyStatusSize = MGPU_ROUND_UP_POW2(sizeof(byte) * numBlocks, 128);

    int total = rangesSize + ranges2Size + mergeListSize + copyListSize +
        countersSize + copyStatusSize;

    support.ranges_global = PtrOffset((int *) mem->get(), 0);
    support.ranges2_global = PtrOffset((int2 *) support.ranges_global, rangesSize);
    support.mergeList_global = PtrOffset((int4 *) support.ranges2_global, ranges2Size);

    support.copyList_global = PtrOffset((int *) support.mergeList_global, mergeListSize);
    support.queueCounters_global = PtrOffset((int2 *) support.copyList_global, copyListSize);
    support.nextCounters_global = PtrOffset(support.queueCounters_global, sizeof(int2));
    support.copyStatus_global = PtrOffset((byte *) support.queueCounters_global, countersSize);

    // Fill the counters with 0s on the first run.
    cudaMemsetAsync(support.queueCounters_global, 0, sizeof(int4), context.Stream());
}


template<MgpuBounds Bounds, typename It1, typename Comp>
void SegmentedSorter::partition
(int count, It1 data_global, int numItems, int nv, Comp comp, CudaContext& context)
{
    const int NT = 64;
    int numBlocks = MGPU_DIV_UP(count, nv);
    int numPartitionBlocks = MGPU_DIV_UP(numBlocks + 1, NT);
    // MGPU_MEM(int) partitionsDevice = context.Malloc<int>(numBlocks + 1);

    KernelBinarySearch<NT, Bounds>
        <<<numPartitionBlocks, NT, 0, context.Stream()>>>
        (count, data_global, numItems, nv, partitionsDevice->get(), numBlocks + 1, comp);
}
