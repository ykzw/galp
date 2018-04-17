// -*- coding: utf-8 -*-

#pragma once

#include "dpp.cuh"
#include "../common/incore.cuh"
#include "../common/sync.cuh"
#include "../common/async.cuh"
#include "../common/hybrid.cuh"


template<typename V, typename E>
using InCoreDPP = InCoreLP<V, E, DPPBase<V, E>>;

template<typename V, typename E>
using SyncDPP = SyncLP<V, E, DPPBase<V, E>>;

template<typename V, typename E>
using AsyncDPP = AsyncLP<V, E, DPPBase<V, E>>;

template<typename V, typename E>
using HybridDPP = HybridLP<V, E, DPPBase<V, E>>;
