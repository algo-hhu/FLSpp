//
// Created by lukas on 21.11.22.
//

// #ifndef FLS___CLUSTERING_H
// #define FLS___CLUSTERING_H

#pragma once
#include <vector>
#include <iterator>
#include <iostream>
#include <cmath>

#include "additional_vector_stuff.h"
#include "makros.h"


class Point
{
public:
    int dimension;
    int index;
    double weight = 1;
    std::vector<double> coordinates;

    Point(int dim, int ind, std::vector<double> coord);
    Point(int dim, int ind, double w, std::vector<double> coord);
    Point(Point const &p);
    Point();

    void operator+=(const Point &other);
};

bool operator==(const Point &lhs, const Point &rhs);
bool operator!=(const Point &lhs, const Point &rhs);

// Overload the += operator to add another Point to this Point

std::ostream &operator<<(std::ostream &os, const Point &p);

double euclidean_distance_squared(Point &x, Point &y);

// #endif //FLS___CLUSTERING_H
