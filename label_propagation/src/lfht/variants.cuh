// -*- coding: utf-8 -*-

#pragma once

#include "lfht.cuh"
#include "../common/incore.cuh"
#include "../common/sync.cuh"
#include "../common/async.cuh"
#include "../common/hybrid.cuh"
#include "../common/multi_async.cuh"
#include "../common/multi_incore.cuh"


template<typename V, typename E>
using InCoreLFHT = InCoreLP<V, E, LFHTBase<V, E>>;

template<typename V, typename E>
using SyncLFHT = SyncLP<V, E, LFHTBase<V, E>>;

template<typename V, typename E>
using AsyncLFHT = AsyncLP<V, E, LFHTBase<V, E>>;

template<typename V, typename E>
using HybridLFHT = HybridLP<V, E, LFHTBase<V, E>>;

template<typename V, typename E>
using MultiAsyncLFHT = MultiAsyncLP<V, E, AsyncLFHT<V, E>>;

// template<typename V, typename E>
// using MultiInCoreLFHT = MultiInCoreLP<V, E, AsyncLFHT<V, E>>;
