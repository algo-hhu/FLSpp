//
// Created by lukas on 01.12.22.
// Implementation of classes defined in clustering.h

#include "clustering.h"

Point::Point(int dim, int ind, std::vector<double> coord) : dimension(dim), index(ind), coordinates(coord)
{
}

Point::Point(Point const &p) : dimension(p.dimension), index(p.index), coordinates(p.coordinates)
{
}

Point::Point()
{
}

bool operator==(const Point &lhs, const Point &rhs)
{
    // we only compare the coordinates, since some of the other values may not be initialized
    return lhs.coordinates == rhs.coordinates;
}

bool operator!=(const Point &lhs, const Point &rhs)
{
    return !(lhs == rhs);
}

std::ostream &operator<<(std::ostream &os, const Point &p)
{
    os << "[ " << p.coordinates << "]";

    return os;
}

void Point::operator+=(const Point &other)
{
    if (coordinates.size() != other.coordinates.size())
    {
        throw std::invalid_argument("Points must have the same number of dimensions.");
    }

    for (size_t i = 0; i < coordinates.size(); ++i)
    {
        coordinates[i] += other.coordinates[i];
    }
}

double euclidean_distance_squared(Point &x, Point &y)
{
    double distance = 0;
    double a, b;
    for (int i = 0; i < x.dimension; ++i)
    {
        // distance += (x.coordinates[i] - y.coordinates[i]) * (x.coordinates[i] - y.coordinates[i]);
        // distance += std::pow(x.coordinates[i] - y.coordinates[i],2);
        a = x.coordinates[i];
        b = y.coordinates[i];
        distance += (a - b) * (a - b);
    }
    return distance;
}

Cluster::Cluster(int ind, Point centr) : index(ind), centroid(centr) {}

void Cluster::add_member(Point p)
{
    members.push_back(p);
}
