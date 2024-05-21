#pragma once

#include <random>
#include <chrono>
#include "makros.h"

template <typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator &g);

template <typename Iter>
Iter select_randomly(Iter start, Iter end);

class SeedGenerator
{
public:
    std::mt19937_64 generateSeed()
    {
        // Use a high-resolution clock to generate a seed
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        return std::mt19937_64(seed);
    }
};

struct RandomGenerator
{
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> unif;

    RandomGenerator(double min = 0, double max = 1, std::size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count()) : rng(seed), unif(min, max) {}
    // RandomGenerator(double min = 0, double max = 1) : rng(SeedGenerator().generateSeed()), unif(min, max) {}
    // RandomGenerator(double min = 0, double max = 1, std::mt19937_64 seed = SeedGenerator().generateSeed()) : rng(seed), unif(min, max) {}
    // RandomGenerator(double min = 0, double max = 1, std::mt19937_64 seed = std::mt19937_64())
    //    : rng((seed == std::mt19937_64()) ? SeedGenerator().generateSeed() : seed), unif(min, max) {}

    double getRandomNumber()
    {
        return unif(rng);
    }
};