#pragma once

#include "clustering.h"
#include "additional_vector_stuff.h"
#include "makros.h"
#include "random_generator.h"
#include <limits>
#include <math.h>
#include <functional>
#include <algorithm> // sort
#include <numeric>

struct output_algorithm
{
	std::vector<Point> final_centers;
	std::vector<int> final_labels;
	double cost;
	int iterations;

public:
	output_algorithm(std::vector<Point> _final_centers, std::vector<int> _final_labels, double _cost, int _iterations) : final_centers{_final_centers}, final_labels{_final_labels}, cost{_cost}, iterations{_iterations} {}

	/* {
		return	final_centers == rhs.final_centers	&&
				final_labels == rhs.final_labels	&&
				cost			== rhs.cost			&&
				iterations		== rhs.iterations;
	}
	*/
	/*
	friend bool operator==(const output_algorithm& lhs, const output_algorithm& rhs) {
		return	lhs.final_centers == rhs.final_centers	&&
				lhs.final_labels	== rhs.final_labels	&&
				lhs.cost			== rhs.cost			&&
				lhs.iterations	== rhs.iterations;
	}
	*/
};

bool operator==(const output_algorithm &lhs, const output_algorithm &rhs);

std::ostream &operator<<(std::ostream &os, const output_algorithm &oa);

// default empty output for algorithm if something went wrong/code is not finished
struct output_empty : output_algorithm
{
public:
	output_empty() : output_algorithm{std::vector<Point>{}, std::vector<int>{}, -1, -1} {}
};

class Clustering_Algorithm
{
public:
	std::vector<Point> points;
	std::vector<Point> centers;
	std::vector<double> closest_center_distances; // vector containing distance to closest center for each point
	std::vector<int> labels;					  // vector containing label of closest center for each point (i.e. cluster membership)
	std::vector<double> cumsums;

	std::size_t seed = 0;

	RandomGenerator unif_generator = RandomGenerator(0, 1, seed);

	double tol_factor = 0.0001;
	double tol;
	int iterations = 0;
	int maximum_number_iterations = -1;

	double get_cost(std::vector<Point> centers);

	/// <summary>MyMethod is a method in the MyClass class.
	/// <para>Here's how you could make a second paragraph in a description. <see cref="System::Console::WriteLine"/> for information about output statements.</para>
	/// <seealso cref="MyClass::MyMethod2"/>
	/// </summary>
	Clustering_Algorithm(std::vector<Point> _points, std::size_t seed = 0, int _max_number_iterations = 100) : points(_points), seed(seed), maximum_number_iterations(_max_number_iterations) {};
	Clustering_Algorithm(std::string filepath, std::size_t seed = 0, char delimiter = ' ', int _max_number_iterations = 100) : seed(seed), maximum_number_iterations(_max_number_iterations){};

	void set_centers(std::vector<Point> _centers) { centers = _centers; }

	void init_values();			  // initialize the centers, distances, labels, cumsum
	virtual bool update_labels(); // update labels to their closest center point
	int choose_center();		  // choose point index according to km++ prob distibrution
	void update_centroids();
	void compute_centroids(std::vector<Point> &new_centers);
	void compute_centroids(const std::vector<int> &input_labels, std::vector<Point> &new_centers); // input labels are not changed so defined as const

	virtual bool brute_force_labels_compare();

	// make class abstract by using pure virtual function
	virtual output_algorithm algorithm(int k, bool init = true, double _old_cost = -1) = 0;
	virtual void initialize_centers(int k) = 0;
};

// for easier reusage of stored information about some clustering
// Here the data is newly allocated
struct information_clustering
{
public:
	std::vector<Point> centers;
	std::vector<double> closest_center_distances;
	std::vector<double> second_closest_center_distances;
	std::vector<int> labels;
	std::vector<int> second_closest_labels;
	std::vector<double> cumsum;

	information_clustering(std::vector<Point> &_centers, std::vector<double> &_closest_center_distances, std::vector<int> &_labels,
						   std::vector<double> &_second_closest_center_distances, std::vector<int> &_second_closest_labels, std::vector<double> &_cumsum) : information_clustering(_closest_center_distances, _labels, _second_closest_center_distances, _second_closest_labels, _cumsum)
	{
		centers = _centers;
	}

	information_clustering(std::vector<double> &_closest_center_distances, std::vector<int> &_labels,
						   std::vector<double> &_second_closest_center_distances, std::vector<int> &_second_closest_labels, std::vector<double> &_cumsum) : closest_center_distances(_closest_center_distances), second_closest_center_distances(_second_closest_center_distances),
																																							labels(_labels), second_closest_labels(_second_closest_labels), cumsum(_cumsum) {}

	information_clustering(int n)
	{
		closest_center_distances.resize(n);
		second_closest_center_distances.resize(n);
		labels.resize(n);
		second_closest_labels.resize(n);
		cumsum.resize(n);
	}

	information_clustering(int n, int k) : information_clustering(n)
	{
		centers.resize(k);
	}

	void set_next_cumsum_value(int index, double value)
	{
		if (index == 0)
			cumsum[index] = value;
		else
			cumsum[index] = cumsum[index - 1] + value;
	}
};

class KMEANS : public Clustering_Algorithm
{

public:
	std::vector<double> second_closest_center_distances;
	std::vector<int> second_closest_labels; // vector containing label of second-closest center for each point

	KMEANS(std::vector<Point> _points, std::size_t seed = 0, int _max_number_iterations = 100) : Clustering_Algorithm(_points, seed, _max_number_iterations) { init_values(); }
	KMEANS(std::string filepath, std::size_t seed = 0, char delimiter = ' ', int _max_number_iterations = 100) : Clustering_Algorithm(filepath, seed, delimiter, _max_number_iterations) { init_values(); }

	bool update_labels();

	void init_values()
	{
		Clustering_Algorithm::init_values();
		second_closest_center_distances = std::vector<double>(points.size(), std::numeric_limits<double>::max());
		second_closest_labels = std::vector<int>(points.size(), -1); // vector containing label of second-closest center for each point
	}

	// baseline distance function (no additional information saved)
	// needs to be static to be passable to get_cost function
	double normal_distance_function(std::vector<Point> &_points, std::vector<Point> &_centers, int _point, int _center);

	// actual definition of kmeans in this class
	output_algorithm algorithm(int k, bool init = true, double _old_cost = -1);

	void initialize_centers(int k);

	bool brute_force_labels_compare(information_clustering &info);

	double get_cost(std::vector<Point> &centers, information_clustering &info, std::function<double(std::vector<Point> &, std::vector<Point> &, int, int)> distance_function);
	double get_cost(std::vector<Point> &centers, information_clustering &info);
	double get_cost() { return cumsums[cumsums.size() - 1]; } // return cost of current clustering, assumes that labels and cumsum is correct

	std::pair<double, std::vector<bool>> get_updated_cost(Point added_center_point); // return cost, if only one single center is added to the set of centers (no update of center labels)
	void update_labels(int added_center, std::vector<bool> &new_closest);

	bool check_break_cond(int iterations, double old_cost, double new_cost, std::vector<Point> old_centers, std::vector<Point> centers);

	bool brute_force_labels_compare() override;
};

class GREEDY_KMEANS : public KMEANS
{

public:
	int z;

	GREEDY_KMEANS(std::vector<Point> _points, std::size_t seed = 0, int _max_number_iterations = 100, int _z = -1) : KMEANS(_points, seed, _max_number_iterations), z(_z) {}
	GREEDY_KMEANS(std::string filepath, std::size_t seed = 0, char delimiter = ' ', int _max_number_iterations = 100, int _z = -1) : KMEANS(filepath, seed, delimiter, _max_number_iterations), z(_z) {}

	bool update_labels() override;
	void update_distances();
	void initialize_centers(int k) override;
	output_algorithm algorithm(int k, bool init = true, double _old_cost = -1) override;

	bool compute_labels_from_given_centroids(std::vector<Point> &new_centroids, std::vector<double> &new_distances, std::vector<int> &new_labels,
											 std::vector<double> &new_second_closest_distances, std::vector<int> &new_second_closest_labels, std::vector<double> &new_cumsums);
	bool compute_labels_from_given_centroids(std::vector<Point> &new_centroids, std::vector<double> &new_distances, std::vector<int> &new_labels,
											 std::vector<double> &new_second_closest_distances, std::vector<int> &new_second_closest_labels, std::vector<double> &new_cumsums, std::vector<double> &new_clustercosts);
};

class LOCAL_SEARCH : public GREEDY_KMEANS
{

public:
	std::vector<std::vector<double>> all_pairwise_distances;

	int local_search_steps = 25;

	std::size_t max_size_points = 1; // if size of pointset is exceeding this value we dont compute all pairwise distances but compute the corresponding distances each time
	bool all_distances_computed = false;

	LOCAL_SEARCH(std::vector<Point> _points, std::size_t seed = 0) : GREEDY_KMEANS(_points, seed) { init_values(); }
	LOCAL_SEARCH(std::string filepath, std::size_t seed = 0, char delimiter = ' ', int _max_number_iterations = 100, int _z = 1, int _local_search_steps = 25) : GREEDY_KMEANS(filepath, seed, delimiter, _max_number_iterations, _z)
	{
		init_values();
		local_search_steps = _local_search_steps;
	}

	double get_cost(std::vector<Point> &centers, std::vector<double> &new_distances, std::vector<int> &new_labels,
					std::vector<double> &new_second_closest_distances, std::vector<int> &new_second_closest_labels, std::vector<double> &new_cumsums);

	void greedy_local_search_center();
	void compute_all_pairwise_distances();
	double get_pointwise_distance(int index1, int index2);

	std::pair<std::vector<int>, std::vector<double>> find_3_closest(int point);

	output_algorithm algorithm(int k, bool init = true, double old_cost = -1);
	// void initialize_centers(int k) override;	// Use same initialize centers function as for greedy kmeans

	// bool update_labels() override;
	void update_labels_initialize_centers();
};

struct FLSPP_configuration
{
	int current_number_consecutive_fails = 0;

public:
	int k = 0;

	// bool standart_run = true;					// run the FLSPP algorithm in default configuration, no improvements in runtime optimization;
	bool decreasing_clustercosts = true;	   // run FLSPP by iteratively considering the costs of each center by respective decreasing clustercost order
	bool first_improve = false;				   // stop to find improvements in local step if one better solution was found
	bool exchange_closest_center = false;	   // select the centers to sampled point in increasing distance order
	bool break_after_successive_fails = false; // stop to find improvements if we fail to find an improvement after specified consecutive number of iterations
											   // need to be careful if number of maximum local search steps iterations is also given since then we could end the loop faster than specified!

	int max_number_fails = 5;

	std::vector<int> exchange_center_order;

	FLSPP_configuration() {}
	FLSPP_configuration(int _k) : k(_k), exchange_center_order(_k) {}

	// helper function for checking if we just want to consider the default ordering
	bool default_order() { return !(decreasing_clustercosts || exchange_closest_center); }

	// helper functions for setting the order of considering the centers
	void set_default_order() { std::iota(exchange_center_order.begin(), exchange_center_order.end(), 0); }
	void set_decreasing_clustercosts_order(std::vector<double> &clustgercosts);

	// helper functions for break and update condition by consecutive number of iterations where no improvement was found
	void update_fail_condition() { current_number_consecutive_fails++; }
	bool check_fail_condition() { return current_number_consecutive_fails >= max_number_fails; }
	void reset_fail_condition() { current_number_consecutive_fails = 0; }
};

class FLSPP : public GREEDY_KMEANS
{

public:
	FLSPP_configuration my_conf;

	int max_number_iterations_foresight = -1; // This value gives us the maximum number of iterations such that we use foresight. Afterwards we continue with lloyds
											  // If set to -1 we will continue until normal convergence criteria, otherwise we always do these number of steps regardless of stopping criteria
	std::vector<double> clustercosts;		  // for better selection strategy

	bool decreasing_clustercosts_ordering = true;

	FLSPP(std::vector<Point> _points, std::size_t seed = 0, int _max_number_iterations = 100, int _z = -1, int _local_search_steps = 20) : GREEDY_KMEANS(_points, seed, _max_number_iterations, _z)
	{
		max_number_iterations_foresight = _local_search_steps; // If heuristic break_after_successive_fails is activated we might terminate faster!
	}

	FLSPP(std::string filepath, std::size_t seed = 0, char delimiter = ' ', int _max_number_iterations = 100, int _z = -1, int _local_search_steps = -1) : GREEDY_KMEANS(filepath, seed, delimiter, _max_number_iterations, _z)
	{
		max_number_iterations_foresight = _local_search_steps; // If heuristic break_after_successive_fails is activated we might terminate faster!
	}

	// init_values() override // already implemented by Greedy_KMEANS
	// void initialize_centers(int k) override; // already implemented by Greedy_KMEANS

	void set_clustercosts();

	bool single_loyds_step(std::vector<Point> &new_centroids, std::vector<double> &new_distances, std::vector<int> &new_labels,
						   std::vector<double> &new_second_closest_distances, std::vector<int> &new_second_closest_labels, std::vector<double> &new_cumsums);
	bool single_loyds_step(std::vector<int> &labels, std::vector<Point> &new_centroids, std::vector<double> &new_distances, std::vector<int> &new_labels,
						   std::vector<double> &new_second_closest_distances, std::vector<int> &new_second_closest_labels, std::vector<double> &new_cumsums);
	bool single_loyds_step(std::vector<int> &labels, std::vector<Point> &new_centroids, std::vector<double> &new_distances, std::vector<int> &new_labels,
						   std::vector<double> &new_second_closest_distances, std::vector<int> &new_second_closest_labels, std::vector<double> &new_cumsums, std::vector<double> &new_clustercosts);
	/*bool single_loyds_step();*/

	void local_search_foresight_iterations(int iterations_foresight); // for doing normal iterations without check for heuristic or convergence criteria

	output_algorithm algorithm(int k, bool init = true, double old_cost = -1) override;
};
