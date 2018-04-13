// -*- coding: utf-8 -*-

#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <tuple>
#include <array>
#include <memory>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <cmath>
#include <chrono>

typedef unsigned int uint;
typedef unsigned long ulong;

// Declarations ________________________________________________________________

// Measure the running time of function 'func' with arguments 'args',
// and print the result and time, and return the time in seconds
template<typename Function, typename... Args>
double measure_function_time(Function func, Args &&... args);

// Return the memoized version of function 'func'
template<typename ReturnType, typename... Args>
std::function<ReturnType (Args...)>
memoize(const std::function<ReturnType (Args...)> &func);

// Return a shared pointer to a vector of prime numbers below 'N'
template<typename IntType>
std::shared_ptr<std::vector<IntType>>
gen_primes(IntType N);

// Return a vector of values of Euler's totient function for 0 < i < N
template<typename IntType> std::shared_ptr<std::vector<IntType>>
comp_totients(IntType N, const std::vector<IntType> &primes);

// Binary search sorted vector 'sv'
template<typename T>
inline bool search_sorted_vector(const std::vector<T> &sv, const T &key);

template<typename OutputIt, typename Size, typename Assignable>
void iota_n(OutputIt first, Size n, Assignable value);

// Binary
template<typename IntType1, typename IntType2>
inline auto gcd(IntType1 x, IntType2 y) -> decltype(x + y);

// n-ary (n > 2)
template<typename IntType1, typename IntType2, typename... Args>
inline auto gcd(IntType1 x, IntType2 y, const Args &... rest) -> decltype(x + y);

// Binary
template<typename T, typename U>
constexpr inline auto max(const T &x, const U &y) -> decltype(x + y);

// n-ary (n > 2)
template<typename T, typename U, typename... Args>
constexpr inline auto max(const T &x, const U &y, const Args &... rest) -> decltype(x + y);

template<typename T>
constexpr inline int sign(T x) { return (T(0) < x) - (x < T(0)); }

// _____________________________________________________________________________


class Timer {
public:
    typedef decltype(std::chrono::system_clock::now()) time_type;

    Timer() {}
    void start() { m_start = my_clock(); }
    void stop() { m_stop = my_clock(); }
    double elapsed_time() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(m_stop - m_start).count() / 1e6;
    }

private:
    time_type m_start, m_stop;
    time_type my_clock() const {
        return std::chrono::system_clock::now();
    }
};


template<typename IntType>
class Range {
public:
    constexpr Range(IntType e): _start(0), _end(e), _step(1) { }
    constexpr Range(IntType s, IntType e): _start(std::min(s, e)), _end(e), _step(1) { }
    // It's invalid to pass arguments such that
    // IntType is unsigned and e < s (i.e., e - s becomes negative)
    constexpr Range(IntType s, IntType e, IntType p):
        _start(s), _end((e - s) / p > 0 ? ((e - s) / p + ((e - s) % p != 0)) * p + s : s), _step(p) { }

    class iterator: public std::iterator<std::input_iterator_tag, IntType> {
    public:
        constexpr iterator(IntType s, const Range &parent): cur(s), parent(parent) { }
        constexpr IntType operator*() const { return cur; }
        iterator &operator++() { cur += parent._step; return *this; };
        const iterator operator++(int) {
            cur += parent._step;
            return iterator(cur - parent._step, parent);
        }

        constexpr bool operator==(const iterator &rhs) const { return cur == rhs.cur; }
        constexpr bool operator!=(const iterator &rhs) const { return cur != rhs.cur; }

    private:
        IntType cur;
        const Range &parent;
    };

    constexpr iterator begin() const { return iterator(_start, *this); }
    constexpr iterator end() const { return iterator(_end, *this); }

private:
    const IntType _start;
    const IntType _end;
    const IntType _step;
};

// Convinience functions for Range
template<typename IntType>
constexpr inline Range<IntType> range(IntType e)
{
    return Range<IntType>(e);
}

template<typename IntType>
constexpr inline Range<IntType> range(IntType s, IntType e)
{
    return Range<IntType>(s, e);
}

template<typename IntType>
constexpr inline Range<IntType> range(IntType s, IntType e, IntType p)
{
    return Range<IntType>(s, e, p);
}


template<typename ReturnType>
struct FunctionTimer {
    template<typename Function, typename... Args>
    static double measure(Function func, Args &&... args) {
        Timer t; t.start();
        ReturnType result = func(std::forward<Args>(args)...);
        t.stop();
        std::cout << "Result: " << result << std::endl
                  << "Elapsed time: " << t.elapsed_time() << " (sec)" << std::endl;
        return t.elapsed_time();
    }
};

template<>
struct FunctionTimer<void> {
    template<typename Function, typename... Args>
    static double measure(Function func, Args &&... args) {
        Timer t; t.start();
        func(std::forward<Args>(args)...);
        t.stop();
        std::cout << "Elapsed time: " << t.elapsed_time() << " (sec)" << std::endl;
        return t.elapsed_time();
    }
};

template<typename Function, typename... Args>
double measure_function_time(Function func, Args &&... args)
{
    typedef decltype(func(args...)) ReturnType;
    return FunctionTimer<ReturnType>::measure(func, std::forward<Args>(args)...);
}



template<typename ReturnType, typename... Args>
std::function<ReturnType (Args...)> memoize(const std::function<ReturnType (Args...)> &func)
{
    typedef std::map<std::tuple<Args...>, ReturnType> cache_type;
    cache_type cache;

    return [func, cache](Args &&... args) mutable -> ReturnType {
        std::tuple<Args...> t(args...);
        ReturnType ret;
        typename cache_type::iterator lb = cache.lower_bound(t);
        if (lb != cache.end() && !(t < lb->first)) {
            ret = cache[t];
        } else {
            ret = func(std::forward<Args>(args)...);
            cache.insert(lb, std::make_pair(t, ret));
            // cache.emplace_hint(lb, t, ret);
        }
        return ret;
    };
}



template<typename IntType>
std::shared_ptr<std::vector<IntType>> gen_primes(IntType N)
{
    std::shared_ptr<std::vector<IntType>> primes(new std::vector<IntType>(1, 2));
    primes->reserve(N * 1.1 / log(N));
    std::vector<bool> sieve(N / 2, true);

    for (auto i: range(1, N / 2)) {
        if (sieve[i]) {
            const IntType p = 2 * i + 1;
            primes->push_back(p);
            for (IntType j = 3 * p; j < N; j += 2 * p) {
                sieve[j / 2] = false;
            }
        }
    }

    return primes;
}


template<typename IntType>
std::shared_ptr<std::vector<IntType>>
comp_totients(IntType N, const std::vector<IntType> &primes)
{
    Range<IntType> r(N);
    std::vector<double> dtotients(r.begin(), r.end());

    for (const IntType p: primes) {
        dtotients[p] = p - 1.0;
        for (std::vector<double>::size_type i = 2 * p; i < N; i += p) {
            dtotients[i] *= 1.0 - 1.0 / p;
        }
    }

    return std::shared_ptr<std::vector<IntType>>(new std::vector<IntType>(dtotients.cbegin(),
                                                                          dtotients.cend()));
}



template<typename T>
inline bool search_sorted_vector(const std::vector<T> &sv, const T &key)
{
    return std::binary_search(sv.cbegin(), sv.cend(), key);
}



template<typename OutputIt, typename Size, typename Assignable>
void iota_n(OutputIt first, Size n, Assignable value)
{
    std::generate_n(first, n, [&value]() {
            return value++;
        });
}


template<typename IntType1, typename IntType2>
inline auto gcd(IntType1 x, IntType2 y) -> decltype(x + y)
{
    decltype(x + y) t;
    while (y) {
        t = x; x = y; y = t % y;
    }
    return x;
}

template<typename IntType1, typename IntType2, typename... Args>
inline auto gcd(IntType1 x, IntType2 y, const Args &... rest) -> decltype(x + y)
{
    return gcd(gcd(x, y), rest...);
}


template<typename T, typename U>
constexpr inline auto max(const T &x, const U &y) -> decltype(x + y)
{
    return x > y ? x : y;
}

template<typename T, typename U, typename... Args>
constexpr inline auto max(const T &x, const U &y, const Args &... rest) -> decltype(x + y)
{
    return max(max(x, y), rest...);
}



template<typename T, typename U>
inline T divup(T x, U y)
{
    return (x + y - 1) / y;
}
