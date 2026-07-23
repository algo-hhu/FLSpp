// #include "clustering_algorithm.h"
// #include "additional_vector_stuff.h"

#include "clustering_algorithm.h"

/*
bool output_algorithm::operator==(const output_algorithm& rhs)
{
	return	final_centers == rhs.final_centers &&
		final_labels == rhs.final_labels &&
		cost == rhs.cost &&
		iterations == rhs.iterations;
}
*/

bool operator==(const output_algorithm &lhs, const output_algorithm &rhs)
{
	return lhs.final_centers == rhs.final_centers &&
		   lhs.final_labels == rhs.final_labels &&
		   lhs.cost == rhs.cost &&
		   lhs.iterations == rhs.iterations;
}

std::ostream &operator<<(std::ostream &os, const output_algorithm &oa)
{
	// std::cout << "centers: " << std::endl;
	// std::cout << oa.final_centers << std::endl << std::endl;

	// std::cout << "labels of points: " << std::endl;
	// std::cout << "(because datasets are often large this is not printed)" << std::endl << std::endl;

	// std::cout << "cost of clustering: " << oa.cost << std::endl << std::endl;
	std::cout << oa.cost << std::endl;
	// std::cout << "number of iterations performing update steps: " << oa.iterations << std::endl;

	return os;
}

bool Clustering_Algorithm::update_labels()
{

	bool change = false;
	double current_min_dist;
	int current_min_label;
	double lastvalue = 0;

	for (std::size_t i = 0; i < points.size(); ++i)
	{
		current_min_dist = std::numeric_limits<double>::max();
		current_min_label = -1;

		for (std::size_t j = 0; j < centers.size(); ++j)
		{
			double dist = euclidean_distance_squared(points[i], centers[j]);
			/*if (dist < closest_center_distances[i]) {
				closest_center_distances[i] = dist;
				labels[i] = j;
			}*/
			if (dist < current_min_dist)
			{
				current_min_dist = dist;
				current_min_label = j;
			}
		}
		if (!change && labels[i] != current_min_label)
			change = true;

		closest_center_distances[i] = current_min_dist;
		labels[i] = current_min_label;

		cumsums[i] = closest_center_distances[i] + lastvalue;
		lastvalue = cumsums[i];
	}

	return change;
}

double Clustering_Algorithm::get_cost(std::vector<Point> _centers)
{
	std::vector<double> _closest_center_distances = std::vector<double>(points.size(), std::numeric_limits<double>::max());
	double _cumsum = 0;

	for (std::size_t i = 0; i < points.size(); ++i)
	{
		for (std::size_t j = 0; j < _centers.size(); ++j)
		{
			double dist = euclidean_distance_squared(points[i], _centers[j]);
			if (dist < _closest_center_distances[i])
			{
				_closest_center_distances[i] = dist;
			}
		}
		_cumsum += _closest_center_distances[i];
	}
	return _cumsum;
}

void Clustering_Algorithm::init_values()
{
	int n = points.size();
	closest_center_distances = std::vector<double>(n, std::numeric_limits<double>::max());
	labels = std::vector<int>(n, -1);
	cumsums = std::vector<double>(n);

	// compute tol as described in Scikit-learn: Machine Learning in Python
	// 1. calculate mean along each coordinate
	int d = points[0].dimension;
	std::vector<double> means(d);
	for (std::size_t i = 0; i < points.size(); i++)
	{
		for (int j = 0; j < d; j++)
		{
			means[j] += points[i].coordinates[j];
		}
	}
	for (int i = 0; i < d; i++)
	{
		means[i] /= points.size();
	}
	// 2. calculate the variance along each axis
	std::vector<double> var(d);
	for (std::size_t i = 0; i < points.size(); i++)
	{
		for (int j = 0; j < d; j++)
		{
			var[j] += (points[i].coordinates[j] - means[j]) * (points[i].coordinates[j] - means[j]);
		}
	}
	for (int i = 0; i < d; i++)
	{
		var[i] /= points.size();
	}
	// 3. calculate the average of the variances and multiply by tolerance factor
	double v = 0;
	for (int i = 0; i < d; i++)
	{
		v += var[i];
	}
	v /= d;
	v *= tol_factor;

	tol = v;
}

/** @brief sample point using D2-sampling
 *
 * @param[in]  void  no parameter
 * @return int return the index of sampled point using D2-sampling
 */
int Clustering_Algorithm::choose_center()
{
	/*
	 * Input: vector of cumulative sums of cost, cost ("potential") of current solution
	 *
	 * Output: index of a point, s.t. each point has probability of being chosen according to D�-sampling
	 */

	// create random integer in range [0, ... ,pot]

	// double randnr = rand() * cumsums.back() / RAND_MAX;
	double randnr = unif_generator.getRandomNumber() * cumsums.back();

	for (std::size_t i = 0; i < cumsums.size(); ++i)
	{
		if (randnr < cumsums[i])
		{
			return i;
		}
	}
	if (cumsums.back() == 0)
	{ // cost can not be reduced, return last point
		return cumsums.size() - 1;
	}
	std::cout << "If this gets printed, the generated number was too big!";
	return cumsums.size() - 1; // failsafe, probably not needed
}

int Clustering_Algorithm::choose_initial_center(std::vector<double> cumulative_weights)
{

	double randnr = unif_generator.getRandomNumber() * cumulative_weights.back();

	for (size_t i = 0; i < cumulative_weights.size(); ++i)
	{
		if (randnr < cumulative_weights[i])
		{
			return i;
		}
	}
	if (cumulative_weights.back() == 0)
	{ // cost can not be reduced, return last point
		return cumulative_weights.size() - 1;
	}
	std::cout << "If this gets printed, the generated number was too big!";
	return cumulative_weights.size() - 1; // failsafe, probably not needed
}

void Clustering_Algorithm::update_centroids()
{
	int dim = points[0].dimension;									  // shorthand for number of coordinates
	std::vector<double> init(dim);									  // just the dim-dimensional 0-vector
	std::vector<std::vector<double>> centroids(centers.size(), init); // empty vector that gets filled with new centroids
	std::vector<double> cluster_weight(centers.size());				  // for each center (cluster), the total weight of the points with that label. Named "weight" rather than "size" because a point of weight w counts as w points; for unweighted input (all weights 1) this is exactly the cluster size

	// initialize vector of k copies of [0,...,0] (which will successively be updated to become the centroids)
	/*  Note that the centroid is the "mean" of a cluster, i.e. sum of all points in the cluster divided by number.
		We do not know cluster size in advance, so we first add up all points in each cluster and keep track of how many
		points were added in which cluster. In the end, every sum of points gets divided by the number of points in the
		respective cluster.
	*/

	/*for (int i = 0; i < centers.size(); ++i) {
		centroids.push_back(init);
	}*/

	// iterate over all points
	for (std::size_t i = 0; i < points.size(); ++i)
	{
		int label = labels[i]; // check to which cluster the current point belongs

		for (int j = 0; j < dim; ++j)
		{
			centroids[label][j] += points[i].coordinates[j]*points[i].weight; // add coordinates of current point to coord. of centroid-to-be, multiplied by weight
		}
		cluster_weight[label] += points[i].weight; // increase weight counter by weight of point, as one point of weight w is treated like w identical points

	}
	for (std::size_t i = 0; i < centroids.size(); ++i)
	{
		for (int j = 0; j < dim; ++j)
		{
			centroids[i][j] = centroids[i][j] / cluster_weight[i]; // divide each coordinate of the summed up vectors by the total weight in resp. cluster
		}
	}

	for (std::size_t i = 0; i < centroids.size(); i++)
	{
		Point new_center(dim, i, centroids[i]);
		centers[i] = new_center;
	}
}

void Clustering_Algorithm::compute_centroids(std::vector<Point> &new_centers)
{
	int dim = points[0].dimension;									  // shorthand for number of coordinates
	std::vector<double> init(dim);									  // just the dim-dimensional 0-vector
	std::vector<std::vector<double>> centroids(centers.size(), init); // empty vector that gets filled with new centroids
	std::vector<double> cluster_weight(centers.size());				  // for each center (cluster), the total weight of the points with that label. Named "weight" rather than "size" because a point of weight w counts as w points; for unweighted input (all weights 1) this is exactly the cluster size

	// initialize vector of k copies of [0,...,0] (which will successively be updated to become the centroids)
	/*  Note that the centroid is the "mean" of a cluster, i.e. sum of all points in the cluster divided by number.
		We do not know cluster size in advance, so we first add up all points in each cluster and keep track of how many
		points were added in which cluster. In the end, every sum of points gets divided by the number of points in the
		respective cluster.
	*/

	/*for (int i = 0; i < centers.size(); ++i) {
		centroids.push_back(init);
	}*/

	// iterate over all points
	for (std::size_t i = 0; i < points.size(); ++i)
	{
		int label = labels[i]; // check to which cluster the current point belongs

		for (int j = 0; j < dim; ++j)
		{
			centroids[label][j] += points[i].coordinates[j] * points[i].weight; // add coordinates of current point to coord. of centroid-to-be, multiplied by weight
		}
		cluster_weight[label] += points[i].weight; // increase weight counter by weight of point (treating point of weight w as w identical points)
	}

	for (std::size_t i = 0; i < centroids.size(); ++i)
	{
		for (int j = 0; j < dim; ++j)
		{
			centroids[i][j] = centroids[i][j] / cluster_weight[i]; // divide each coordinate of the summed up vectors by the total weight in resp. cluster
		}
	}

	for (std::size_t i = 0; i < centroids.size(); i++)
	{
		Point new_center(dim, i, centroids[i]);
		new_centers[i] = new_center;
	}
}

// TODO make simpler for call without input labels
void Clustering_Algorithm::compute_centroids(std::vector<int> const &input_labels, std::vector<Point> &new_centers)
{
	int dim = points[0].dimension;									  // shorthand for number of coordinates
	std::vector<double> init(dim);									  // just the dim-dimensional 0-vector
	std::vector<std::vector<double>> centroids(centers.size(), init); // empty vector that gets filled with new centroids
	std::vector<double> cluster_weight(centers.size());				  // for each center (cluster), the total weight of the points with that label. Named "weight" rather than "size" because a point of weight w counts as w points; for unweighted input (all weights 1) this is exactly the cluster size

	// initialize vector of k copies of [0,...,0] (which will successively be updated to become the centroids)
	/*  Note that the centroid is the "mean" of a cluster, i.e. sum of all points in the cluster divided by number.
		We do not know cluster size in advance, so we first add up all points in each cluster and keep track of how many
		points were added in which cluster. In the end, every sum of points gets divided by the number of points in the
		respective cluster.
	*/

	std::size_t i;
	int j, label;

	// iterate over all points
	for (i = 0; i < points.size(); ++i)
	{
		label = input_labels[i]; // check to which cluster the current point belongs

		for (j = 0; j < dim; ++j)
		{
			centroids[label][j] += points[i].coordinates[j]*points[i].weight; // add (weighted) coordinates of current point to coord. of centroid-to-be
		}
		cluster_weight[label] += points[i].weight; // increase weight counter by weight of point (treating point of weight w as w identical points)
	}

	for (i = 0; i < centroids.size(); ++i)
	{
		for (j = 0; j < dim; ++j)
		{
			centroids[i][j] = centroids[i][j] / cluster_weight[i]; // divide each coordinate of the summed up vectors by the total weight in resp. cluster
		}
	}

	for (std::size_t i = 0; i < centroids.size(); i++)
	{
		Point new_center(dim, i, centroids[i]);
		new_centers[i] = new_center;
	}
}

bool Clustering_Algorithm::brute_force_labels_compare()
{
#ifndef DEBUG
	return true;
#endif // !DEBUG

	std::vector<double> new_cumsum(points.size());
	for (std::size_t i = 0; i < points.size(); i++)
	{
		if (euclidean_distance_squared(points[i], centers[labels[i]]) != closest_center_distances[i])
		{
			std::cout << "distance of point to its label center is not correctly saved in closest_center_distances: " << std::endl;
			std::cout << "closest_center_distances[" << i << "] = " << closest_center_distances[i] << " , but computed value " << euclidean_distance_squared(points[i], centers[labels[i]]) << std::endl;
			return false;
		}
		// compare currently saved minimum distance and label to all pairwise centers
		double computed_closest_distance = std::numeric_limits<double>::max();
		double computed_second_closest_distance = std::numeric_limits<double>::max();

		// iterate through every point and center, check if current set labels and distances are correct
		for (std::size_t j = 0; j < centers.size(); j++)
		{
			double new_distance = euclidean_distance_squared(points[i], centers[j]);
			if (new_distance <= computed_closest_distance)
			{
				computed_second_closest_distance = computed_closest_distance;
				computed_closest_distance = new_distance;
			}
			else if (new_distance < computed_second_closest_distance)
			{
				computed_second_closest_distance = new_distance;
			}
		}

		// check if distance of current labels is the same
		if (computed_closest_distance != closest_center_distances[i])
		{
			std::cout << "point " << i << ":" << std::endl;
			std::cout << "distance of closest center is not correct: found " << closest_center_distances[i] << " but computed " << computed_closest_distance << std::endl;
			return false;
		}

		// check cumsum values

		double new_cumsum_value = closest_center_distances[i];
		if (i == 0)
		{
			if (new_cumsum_value != cumsums[i])
			{
				std::cout << "first cumsum value is wrong: expected " << new_cumsum_value << " , but got " << cumsums[i] << std::endl;
			}
			else
				new_cumsum[i] = new_cumsum_value;
		}
		else
		{
			if (new_cumsum[i - 1] + new_cumsum_value != cumsums[i])
			{
				std::cout << "cumsum value is wrong: expected " << new_cumsum[i - 1] + new_cumsum_value << " , but got " << cumsums[i] << std::endl;
			}
			else
			{
				new_cumsum[i] = new_cumsum[i - 1] + new_cumsum_value;
			}
		}
	}

	return true;
}

/** @brief kmeans algorithm
 *
 * function longer description if need
 * @param[in]  void  no parameter
 * @param[in]  param2_name  description of parameter2
 * @return output_algorithm return the results of converged kmeans
 */
output_algorithm KMEANS::algorithm(int k, bool init, double _old_cost)
{
	// First we use some standard initialization method (could be made more generally as an additional parameter)
	if (init)
		initialize_centers(k);

	// std::vector<int> old_labels(points.size());
	// old_labels[0] = -1;

	double old_cost = _old_cost;
	double new_cost;
	bool change = true;

	// std::cout << "initialization cost: " << cumsums.back() << std::endl;

	std::vector<Point> old_centers;

	while (change)
	{
		// break condition because of maximum number of iterations was reached
		if (iterations == maximum_number_iterations)
			break;

		// old_labels = labels;
		old_centers = centers;

		update_centroids();
		change = update_labels();
		brute_force_labels_compare();
		// std::cout << "current cost: " << cumsums.back() << std::endl;

		new_cost = cumsums.back();

		// specific break condition like in master thesis
		bool break_cond = check_break_cond(iterations, old_cost, new_cost, old_centers, centers);
		if (break_cond)
			break;

		iterations++;
		old_cost = new_cost;
	}

	return output_algorithm(centers, labels, cumsums.back(), iterations);
}

double KMEANS::normal_distance_function(std::vector<Point> &_points, std::vector<Point> &_centers, int _point, int _center)
{
	return euclidean_distance_squared(_points[_point], _centers[_center]);
}

void KMEANS::initialize_centers(int k)
{

	// first select the first center uniformly at random
	int randnr = (int)(unif_generator.getRandomNumber() * points.size());

	// if centers are already initialized reset to size 0
	if (centers.size() > 0)
		centers.resize(0);

	centers.push_back(points[randnr]);

	update_labels();

	// now we select the following centers proportional to the probability of their costs
	while (static_cast<int>(centers.size()) < k)
	{
		int new_center = choose_center();

		centers.push_back(points[new_center]);

		update_labels();
	}
}


/** @brief computes the cost of some given center set using a self defined distance function
 *
 * @param[in]  centers  set of centers for current clustering
 * @param[in]  information_clustering  preinitialized struct to contain informations afterwards
 * @param[in]  distance_function  distance function to compute distance between points[i] and centers[j]
 * @return output_algorithm return the results of converged kmeans
 */
double KMEANS::get_cost(std::vector<Point> &_centers, information_clustering &info, std::function<double(std::vector<Point> &, std::vector<Point> &, int, int)> distance_function)
{
	for (std::size_t i = 0; i < points.size(); ++i)
	{
		double current_min_dist = std::numeric_limits<double>::max();
		int current_min_dist_label = -1;
		double current_second_min_dist = std::numeric_limits<double>::max();
		int current_second_min_dist_label = -1;

		// iterate through every center combination and update closest and second-closest
		for (std::size_t j = 0; j < centers.size(); ++j)
		{
			// double dist = euclidean_distance_squared(points[i], new_centroids[j]);
			double dist = distance_function(points, _centers, i, j);

			if (dist < current_min_dist)
			{
				current_second_min_dist = current_min_dist;
				current_second_min_dist_label = current_min_dist_label;

				current_min_dist = dist;
				current_min_dist_label = j;
			}
			else if (dist < current_second_min_dist)
			{
				current_second_min_dist = dist;
				current_second_min_dist_label = j;
			}
		}

		// set information about clustering in struct
		info.closest_center_distances[i] = current_min_dist;
		info.labels[i] = current_min_dist_label;
		info.second_closest_center_distances[i] = current_second_min_dist;
		info.second_closest_labels[i] = current_second_min_dist_label;
		info.set_next_cumsum_value(i, current_min_dist, points[i].weight);
	}
	return info.cumsum.back();
}

double KMEANS::get_cost(std::vector<Point> &centers, information_clustering &info)
{
	// auto my_dist_func = [](int point, int center) { return normal_distance_function(point, center); };
	return get_cost(centers, info, ([&](std::vector<Point> &_points, std::vector<Point> &_centers, int _point, int _center) -> double
									{ return normal_distance_function(_points, _centers, _point, _center); }));
}

/** @brief computes the cost of our clutering if only one center point is added to the solution and which points should be assigned to the new center
 *
 * @param[in]  added_center_point  point which gets added to the current clustering
 * @return std::pair<new_cost, new_closest_change> return the cost of new clustering and bool vector which points should now be assigned to the new center
 */
std::pair<double, std::vector<bool>> KMEANS::get_updated_cost(Point added_center_point)
{
	// we iterate through all points and compare with the cost of adding some center point
	double new_cost = 0;
	double new_dist = 0;
	std::vector<bool> new_closest_change(points.size());

	for (std::size_t i = 0; i < points.size(); i++)
	{
		new_dist = euclidean_distance_squared(points[i], added_center_point);
		if (closest_center_distances[i] < new_dist)
		{
			new_cost += closest_center_distances[i]*points[i].weight;
		}
		else
		{
			new_cost += new_dist*points[i].weight;
			new_closest_change[i] = true;
		}
	}
	return std::make_pair(new_cost, new_closest_change);
}

/** @brief updates the current labels, secondclosest distances/labels and cumsum given the information which point was added to the clustering and for which points the added_center is now closest center
 *
 * @param[in]  added_centers  index of point added to the clustering
 * @param[in]  new_closest  set of bools such that for every point i we have new_closest[i] = true <=> point i new closest center changes to added_center
 * @return void
 */
void KMEANS::update_labels(int added_center, std::vector<bool> &new_closest)
{
	double new_dist = 0;
	for (std::size_t i = 0; i < points.size(); i++)
	{
		new_dist = euclidean_distance_squared(points[i], centers[added_center]);

		if (new_closest[i])
		{ // if new_closest[i] = true, the added_center becomes closest and old closest becomes secondclosest
			second_closest_labels[i] = labels[i];
			second_closest_center_distances[i] = closest_center_distances[i];
			labels[i] = added_center;
			closest_center_distances[i] = new_dist;
			if (i == 0)
			{
				cumsums[i] = new_dist * points[i].weight;
			}
			else
			{
				cumsums[i] = cumsums[i - 1] + new_dist * points[i].weight;
			}
		}
		else
		{ // otherwise we check if added_center becomes new second_closest
			if (new_dist < second_closest_center_distances[i])
			{
				second_closest_labels[i] = added_center;
				second_closest_center_distances[i] = new_dist;
			}
			if (i == 0)
			{
				cumsums[i] = closest_center_distances[i] * points[i].weight;
			}
			else
			{
				cumsums[i] = cumsums[i - 1] + closest_center_distances[i] * points[i].weight;
			}
		}
	}
}

bool KMEANS::check_break_cond(int iterations, double old_cost, double new_cost, std::vector<Point> old_centers, std::vector<Point> centers)
{
	// if we have first iteration or undefined old cost we continue in any case

	if (iterations == 0 || old_cost == -1)
		return false;

	// old variant for break condituen
	if (1 - new_cost / old_cost < tol_factor)
		return true;

	// new condition: we compute the norm of the difference of the centers and check if this is smaller than tol

	// double norm_val = 0;
	// for (int i = 0; i < old_centers.size(); i++) {
	//	double current_val = 0;
	//	Point difference_centers(old_centers[i]);
	//	for (int j = 0; j < old_centers[0].dimension; j++) {
	//		difference_centers.coordinates[j] -= centers[i].coordinates[j];
	//		current_val += difference_centers.coordinates[j] * difference_centers.coordinates[j];
	//	}
	//	norm_val += current_val;
	// }
	// norm_val = std::sqrt(norm_val);

	//// Problem: If center was exchanged this makes the result not fitting

	// if (norm_val < tol) return true;

	return false;
}

/** @brief brute force comparison
 *
 * compute all distances of closest and secondclosest and check if current labels and saved distances fit.
 * @param[in]  void  no parameter
 * @return bool  return true if same output as currently in object, otherwise print message to stdcout and return false
 */
bool KMEANS::brute_force_labels_compare()
{
#ifndef DEBUG
	return true;
#endif // !DEBUG

	std::vector<double> new_cumsum(points.size());
	for (std::size_t i = 0; i < points.size(); i++)
	{
		if (euclidean_distance_squared(points[i], centers[labels[i]]) != closest_center_distances[i])
		{
			std::cout << "distance of point to its label center is not correctly saved in closest_center_distances: " << std::endl;
			std::cout << "closest_center_distances[" << i << "] = " << closest_center_distances[i] << " , but computed value " << euclidean_distance_squared(points[i], centers[labels[i]]) << std::endl;
			return false;
		}
		if (second_closest_labels[i] != -1 && euclidean_distance_squared(points[i], centers[second_closest_labels[i]]) != second_closest_center_distances[i])
		{
			std::cout << "distance of point to its secondclosest label center is not correctly saved in second_closest_center_distances: " << std::endl;
			std::cout << "second_closest_center_distances[" << i << "] = " << second_closest_center_distances[i] << " , but computed value " << euclidean_distance_squared(points[i], centers[second_closest_labels[i]]) << std::endl;
			return false;
		}
		// compare currently saved minimum distance and label to all pairwise centers
		double computed_closest_distance = std::numeric_limits<double>::max();
		double computed_second_closest_distance = std::numeric_limits<double>::max();
		int closest_index = -1;
		int second_closest_index = -1;

		// iterate through every point and center, check if current set labels and distances are correct
		for (std::size_t j = 0; j < centers.size(); j++)
		{
			double new_distance = euclidean_distance_squared(points[i], centers[j]);
			if (new_distance <= computed_closest_distance)
			{
				computed_second_closest_distance = computed_closest_distance;
				second_closest_index = closest_index;

				computed_closest_distance = new_distance;
				closest_index = j;
			}
			else if (new_distance < computed_second_closest_distance)
			{
				computed_second_closest_distance = new_distance;
				second_closest_index = j;
			}
		}

		// check if distance of current labels is the same
		if (computed_closest_distance != closest_center_distances[i])
		{
			std::cout << "point " << i << ":" << std::endl;
			std::cout << "distance of closest center is not correct: found " << closest_center_distances[i] << " but computed " << computed_closest_distance << std::endl;
			std::cout << "index of closest real center: " << closest_index << std::endl;
			return false;
		}
		if (computed_second_closest_distance != second_closest_center_distances[i])
		{
			std::cout << "point " << i << ":" << std::endl;
			std::cout << "distance of secondclosest center is not correct: found " << second_closest_center_distances[i] << " but computed " << computed_second_closest_distance << std::endl;
			std::cout << "index of second closest real center: " << second_closest_index << std::endl;
			return false;
		}

		// check cumsum values

		double new_cumsum_value = closest_center_distances[i];
		if (i == 0)
		{
			if (new_cumsum_value != cumsums[i])
			{
				std::cout << "first cumsum value is wrong: expected " << new_cumsum_value << " , but got " << cumsums[i] << std::endl;
			}
			else
				new_cumsum[i] = new_cumsum_value;
		}
		else
		{
			if (new_cumsum[i - 1] + new_cumsum_value != cumsums[i])
			{
				std::cout << "cumsum value is wrong: expected " << new_cumsum[i - 1] + new_cumsum_value << " , but got " << cumsums[i] << std::endl;
			}
			else
			{
				new_cumsum[i] = new_cumsum[i - 1] + new_cumsum_value;
			}
		}
	}

	return true;
}

// same as for greedy
bool KMEANS::update_labels()
{
	bool change = false;

	for (std::size_t i = 0; i < points.size(); ++i)
	{
		double current_min_dist = std::numeric_limits<double>::max();
		int current_min_dist_label = -1;
		double current_second_min_dist = std::numeric_limits<double>::max();
		int current_second_min_dist_label = -1;

		// iterate through every center combination and update closest and second-closest
		for (std::size_t j = 0; j < centers.size(); ++j)
		{
			double dist = euclidean_distance_squared(points[i], centers[j]);

			if (dist < current_min_dist)
			{
				current_second_min_dist = current_min_dist;
				current_second_min_dist_label = current_min_dist_label;

				current_min_dist = dist;
				current_min_dist_label = j;
			}
			else if (dist < current_second_min_dist)
			{
				current_second_min_dist = dist;
				current_second_min_dist_label = j;
			}
		}

		if (!change && current_min_dist_label != labels[i])
			change = true;

		labels[i] = current_min_dist_label;
		closest_center_distances[i] = current_min_dist;

		second_closest_labels[i] = current_second_min_dist_label;
		second_closest_center_distances[i] = current_second_min_dist;

		if (i == 0)
		{
			cumsums[i] = closest_center_distances[i]*points[i].weight;
		}
		else
		{
			cumsums[i] = cumsums[i - 1] + closest_center_distances[i]*points[i].weight;
		}
	}

	return change;
}

/** @brief kmeans algorithm
 *
 * function longer description if need
 * @param[in]  void  no parameter
 * @param[in]  param2_name  description of parameter2
 * @return output_algorithm return the results of converged kmeans
 */
output_algorithm GREEDY_KMEANS::algorithm(int k, bool init, double _old_cost)
{
	// First we use some standard initialization method (could be made more generally as an additional parameter)
	if (init)
		initialize_centers(k);

	bool change = true;

	double old_cost = _old_cost;
	double new_cost;

	std::vector<Point> old_centers;

	while (change)
	{
		if (iterations == maximum_number_iterations)
			break;

		old_centers = centers;

		update_centroids();
		change = update_labels();
		brute_force_labels_compare();

		new_cost = cumsums.back();

		bool break_cond = check_break_cond(iterations, old_cost, new_cost, old_centers, centers);
		if (break_cond)
			break;

		old_cost = new_cost;
		iterations++;
	}

	return output_algorithm(centers, labels, cumsums.back(), iterations);
}

void GREEDY_KMEANS::update_distances()
{
	// using current labels update the distances
	for (std::size_t i = 0; i < points.size(); i++)
	{
		closest_center_distances[i] = euclidean_distance_squared(points[i], centers[labels[i]]);
		if (second_closest_labels[i] != -1)
		{
			second_closest_center_distances[i] = euclidean_distance_squared(points[i], centers[second_closest_labels[i]]);
		}
		if (i == 0)
		{
			cumsums[i] = closest_center_distances[i]*points[i].weight;
		}
		else
		{
			cumsums[i] = cumsums[i - 1] + closest_center_distances[i]*points[i].weight;
		}
	}
}

void GREEDY_KMEANS::initialize_centers(int k)
{
	if (z == -1)
		z = 2 + (int)(log(k));

	// if centers are already initialized reset to size 0
	if (centers.size() > 0)
		centers.resize(0);

	// Compute sum of weights of whole point set. If a point has high weight, we want to sample it w/ higher probability
	std::vector<double> cumulative_weights(points.size());
	cumulative_weights[0] = points[0].weight;

	for (size_t i = 1; i < points.size(); ++i)
	{
		cumulative_weights[i] = cumulative_weights[i - 1] + points[i].weight;
	}

	// choose initial center w.r.t sample weights
	int initial_center = choose_initial_center(cumulative_weights);
	centers.push_back(points[initial_center]);

	update_labels();

	double current_cost = cumsums.back();
	double new_cost;

	int best_rand = initial_center;

	// for our current best information we copy our current set in best_information and create an empty set in new_information
	information_clustering best_information(closest_center_distances, labels, second_closest_center_distances, second_closest_labels, cumsums);
	information_clustering new_information(points.size());

	// first select the first center uniformly at random
	for (int i = 1; i < z; i++)
	{

		int new_rand = std::min((int)(points.size() - 1), (int)(unif_generator.getRandomNumber() * points.size()));

		std::vector<Point> new_clustering{points[new_rand]};  // clustering only consists of the single sampled point
		new_cost = get_cost(new_clustering, new_information); // use normal cost-function from kmeans class and normal distance function

		if (new_cost < current_cost)
		{
			best_information = new_information; // overwrites old best informations which in the beginning is just a copy of the original informations
			current_cost = new_cost;
			best_rand = new_rand;
		}
	}

	// using the best_information we update our clustering
	centers[0] = points[best_rand];
	closest_center_distances = best_information.closest_center_distances;
	second_closest_labels = best_information.second_closest_labels;
	cumsums = best_information.cumsum;

	// now we select the following centers proportional to the probability of their costs
	while (static_cast<int>(centers.size()) < k)
	{
		std::vector<int> candidates(z);
		int best_candidate = 0;

		std::pair<double, std::vector<bool>> updated_cost; // contains the new cost of the solution and a bool vector which points are assigned to this new center as closest
		std::pair<double, std::vector<bool>> best_updated_cost;
		std::vector<bool> best_closest_change;
		int new_center;

		for (int i = 0; i < z; i++)
		{
			new_center = choose_center();

			updated_cost = get_updated_cost(points[new_center]);

			if (i == 0 || updated_cost.first < current_cost)
			{
				best_candidate = new_center;
				current_cost = updated_cost.first;
				best_closest_change = updated_cost.second;
			}
		}

		centers.push_back(points[best_candidate]);

		// update_labels();
		KMEANS::update_labels(centers.size() - 1, best_closest_change); // update labels and remaining informations using precomputed information for which points the new candidate becomes closest
	}
}

// return true if label in at least one point did change
bool GREEDY_KMEANS::update_labels()
{
	bool change = false;

	for (std::size_t i = 0; i < points.size(); ++i)
	{
		double current_min_dist = std::numeric_limits<double>::max();
		int current_min_dist_label = -1;
		double current_second_min_dist = std::numeric_limits<double>::max();
		int current_second_min_dist_label = -1;

		// iterate through every center combination and update closest and second-closest
		for (std::size_t j = 0; j < centers.size(); ++j)
		{
			double dist = euclidean_distance_squared(points[i], centers[j]);

			if (dist < current_min_dist)
			{
				current_second_min_dist = current_min_dist;
				current_second_min_dist_label = current_min_dist_label;

				current_min_dist = dist;
				current_min_dist_label = j;
			}
			else if (dist < current_second_min_dist)
			{
				current_second_min_dist = dist;
				current_second_min_dist_label = j;
			}
		}

		if (!change && current_min_dist_label != labels[i])
			change = true;

		labels[i] = current_min_dist_label;
		closest_center_distances[i] = current_min_dist;

		second_closest_labels[i] = current_second_min_dist_label;
		second_closest_center_distances[i] = current_second_min_dist;

		if (i == 0)
		{
			cumsums[i] = closest_center_distances[i] * points[i].weight;
		}
		else
		{
			cumsums[i] = cumsums[i - 1] + closest_center_distances[i] * points[i].weight;
		}
	}

	return change;
}

bool GREEDY_KMEANS::compute_labels_from_given_centroids(std::vector<Point> &new_centroids, std::vector<double> &new_distances, std::vector<int> &new_labels,
														std::vector<double> &new_second_closest_distances, std::vector<int> &new_second_closest_labels, std::vector<double> &new_cumsums)
{
	bool change = false;

	for (std::size_t i = 0; i < points.size(); ++i)
	{
		double current_min_dist = std::numeric_limits<double>::max();
		int current_min_dist_label = -1;
		double current_second_min_dist = std::numeric_limits<double>::max();
		int current_second_min_dist_label = -1;

		// iterate through every center combination and update closest and second-closest
		for (std::size_t j = 0; j < new_centroids.size(); ++j)
		{
			double dist = euclidean_distance_squared(points[i], new_centroids[j]);

			if (dist < current_min_dist)
			{
				current_second_min_dist = current_min_dist;
				current_second_min_dist_label = current_min_dist_label;

				current_min_dist = dist;
				current_min_dist_label = j;
			}
			else if (dist < current_second_min_dist)
			{
				current_second_min_dist = dist;
				current_second_min_dist_label = j;
			}
		}

		if (!change && current_min_dist_label != labels[i])
			change = true;

		new_labels[i] = current_min_dist_label;
		new_distances[i] = current_min_dist;

		new_second_closest_labels[i] = current_second_min_dist_label;
		new_second_closest_distances[i] = current_second_min_dist;

		if (i == 0)
		{
			new_cumsums[i] = new_distances[i] * points[i].weight;
		}
		else
		{
			new_cumsums[i] = new_cumsums[i - 1] + new_distances[i] * points[i].weight;
		}
	}

	return change;
}

bool GREEDY_KMEANS::compute_labels_from_given_centroids(std::vector<Point> &new_centroids, std::vector<double> &new_distances, std::vector<int> &new_labels,
														std::vector<double> &new_second_closest_distances, std::vector<int> &new_second_closest_labels, std::vector<double> &new_cumsums, std::vector<double> &new_clustercosts)
{
	bool change = false;

	for (std::size_t i = 0; i < points.size(); ++i)
	{
		double current_min_dist = std::numeric_limits<double>::max();
		int current_min_dist_label = -1;
		double current_second_min_dist = std::numeric_limits<double>::max();
		int current_second_min_dist_label = -1;

		// iterate through every center combination and update closest and second-closest
		for (std::size_t j = 0; j < new_centroids.size(); ++j)
		{
			double dist = euclidean_distance_squared(points[i], new_centroids[j]);

			if (dist < current_min_dist)
			{
				current_second_min_dist = current_min_dist;
				current_second_min_dist_label = current_min_dist_label;

				current_min_dist = dist;
				current_min_dist_label = j;
			}
			else if (dist < current_second_min_dist)
			{
				current_second_min_dist = dist;
				current_second_min_dist_label = j;
			}
		}

		if (!change && current_min_dist_label != labels[i])
			change = true;

		new_labels[i] = current_min_dist_label;
		new_distances[i] = current_min_dist;

		new_clustercosts[current_min_dist_label] += current_min_dist; // for the closest center we add the found smallest distance to its clustercost

		new_second_closest_labels[i] = current_second_min_dist_label;
		new_second_closest_distances[i] = current_second_min_dist;

		if (i == 0)
		{
			new_cumsums[i] = new_distances[i];
		}
		else
		{
			new_cumsums[i] = new_cumsums[i - 1] + new_distances[i];
		}
	}

	return change;
}

// ------------------------------------------- FLS++ -----------------------------------------------------

void FLSPP::local_search_foresight_iterations(int iterations_foresight)
{

	std::vector<Point> centroids_best(centers.size());
	std::vector<double> closest_center_distances_best(points.size());
	std::vector<int> closest_center_distances_labels_best(points.size());
	std::vector<double> second_closest_center_distances_best(points.size());
	std::vector<int> second_closest_center_distances_labels_best(points.size());
	std::vector<double> new_cumsums_best(points.size());
	std::vector<double> new_clustercosts_best(centers.size());

	// std::vector<Point> centroids_new(centers.size());
	std::vector<double> closest_center_distances_changed(points.size());
	std::vector<int> closest_center_distances_changed_labels(points.size());
	std::vector<double> second_closest_center_distances_changed(points.size());
	std::vector<int> second_closest_center_distances_changed_labels(points.size());
	std::vector<double> new_cumsums(points.size());
	std::vector<double> new_clustercosts(centers.size());

	std::vector<Point> new_centroids(centers.size());
	std::vector<double> new_distances(points.size());
	std::vector<int> new_labels(points.size());

	for (int it = 0; it < iterations_foresight; it++)
	{
		/*
		c_1(p) := closest center to p
		c_2(p) := second closest center to p

		1) Sample point c' using D2 (requires information about closest distances of points to centers
		2) Compute distances d(p,c') for all p e P
		3) Check if some center c e C should be swapped with c' in O(nk):
			- if c_1(p) != c' : compare d(p, c') with d(p, c_1(p)) [O(1)]
			- otherwise		  : compare d(p, c') with d(p, c_2(p)) [O(1)]
			Use information to find best swap candidate c' (or dont swap)
		4) Recompute centroids and memberships to closest and secondclosest
		*/

		// my_conf.update();

		// try closest/smallest cost center?

		// sample new center candidate by current cumsum distribution
		int center_candidate = choose_center();

		// Compute the distances of sampled point to every other point. Could be precomputed but cost decrease might be most likely worse
		for (std::size_t i = 0; i < points.size(); i++)
		{
			new_distances[i] = euclidean_distance_squared(points[i], points[center_candidate]);
		}

		// compute cost of one loyds step using same set of centers, save solution
		/*change_best = single_loyds_step(labels, centroids_best, closest_center_distances_best, closest_center_distances_labels_best,
			second_closest_center_distances_best, second_closest_center_distances_labels_best, new_cumsums_best, new_clustercosts_best);*/
		compute_centroids(labels, centroids_best);
		double new_cost = 0;
		for (std::size_t b = 0; b < points.size(); b++)
		{
			new_cost += euclidean_distance_squared(points[b], centroids_best[labels[b]]) * points[b].weight;
		}

		// set starting cost of solution
		// double current_min_cost = new_cumsums_best.back();
		double current_min_cost = new_cost;
		// int best_change = -1;

		// for (int l = 0; l < my_conf.exchange_center_order.size(); l++) { // we use self defined ordering

		// we iterate over every possible exchange situation and save the best choice
		for (int i = 0; i < static_cast<int>(centers.size()); i++)
		{ // since we always do these number of iterations we do not care about the order of exchanges
			for (std::size_t j = 0; j < points.size(); j++)
			{
				if (labels[j] != i)
				{ // center was not exchanged
					if (closest_center_distances[j] <= new_distances[j])
					{
						new_labels[j] = labels[j];
					}
					else
					{
						new_labels[j] = i;
					}
				}
				else
				{
					if (second_closest_center_distances[j] <= new_distances[j])
					{
						new_labels[j] = second_closest_labels[j];
					}
					else
					{
						new_labels[j] = i;
					}
				}
			}

			compute_centroids(new_labels, new_centroids);
			double new_cost = 0;
			for (std::size_t b = 0; b < points.size(); b++)
			{
				new_cost += euclidean_distance_squared(points[b], new_centroids[new_labels[b]]) * points[b].weight;
			}

			// compare cost of clustering to current best, overwrite if better result
			if (current_min_cost > new_cost)
			{

				centroids_best = new_centroids;
				current_min_cost = new_cost;
				// best_change = i;
			}
		}

		// Update information (this could be modified to assign pointers)
		/*centers = centroids_best;
		closest_center_distances = closest_center_distances_best;
		labels = closest_center_distances_labels_best;
		second_closest_center_distances = second_closest_center_distances_best;
		second_closest_labels = second_closest_center_distances_labels_best;
		cumsums = new_cumsums_best;*/

		centers = centroids_best;
		compute_labels_from_given_centroids(centroids_best, closest_center_distances, labels, second_closest_center_distances, second_closest_labels, cumsums);

		// check for correctness (if makro is defined)
		brute_force_labels_compare();

		// update new cost and check if stop condition factor is reached
		// new_cost = cumsums.back();
	}
}

/** @brief kmeans algorithm
 *
 * function longer description if need
 * @param[in]  void  no parameter
 * @param[in]  param2_name  description of parameter2
 * @return output_algorithm return the results of converged kmeans
 */
output_algorithm FLSPP::algorithm(int k, bool init, double _old_cost)
{

	// First we use some standard initialization method (could be made more generally as an additional parameter)
	if (init)
	{
		initialize_centers(k);
	}

	// From the sampled centers we first compute the centroids, i.e., we do one lloyd step
	update_centroids();
	update_labels();

	// Do specified number of flspp iterations
	local_search_foresight_iterations(max_number_iterations_foresight);

	// We continue with standard Lloyds implementation without setting the centers in a new way
	return GREEDY_KMEANS::algorithm(k, false, _old_cost);
}

/** @brief single loyds step
 *
 * compute new entries after 1 loyds step (requires that current object closest_distances/labels are correct)
 * @param[in]  new_distances  vector of new distances after loyds step (must be initialized!)
 * @param[in]  new_labels  vector of labels of points to closest center  (must be initialized!)
 * @param[in]  new_cumsums vector of new cumsums  (must be initialized!)
 * @return void fill out specified vectors for 1 loyds step
 */
bool FLSPP::single_loyds_step(std::vector<Point> &new_centroids, std::vector<double> &new_distances, std::vector<int> &new_labels,
							  std::vector<double> &new_second_closest_distances, std::vector<int> &new_second_closest_labels, std::vector<double> &new_cumsums)
{
	// compute new centroids
	compute_centroids(new_centroids);

	// compute new labels/distances/ cumsum
	bool change = compute_labels_from_given_centroids(new_centroids, new_distances, new_labels, new_second_closest_distances, new_second_closest_labels, new_cumsums);

	return change;
}

/** @brief single loyds step
 *
 * compute new entries after 1 loyds step. Here we also specify some labeling and output the new cenroids/labels etc.
 * @param[in]	labels	(closest) labels of all points in current assignment
 * @param[in]	new_centroids	vector of new centers after 1 loyd step (centroid computation of labels, must be initialized!)
 * @param[in]	new_distances  vector of new distances after loyds step (must be initialized!)
 * @param[in]	new_labels  vector of labels of points to closest center  (must be initialized!)
 * @param[in]	new_cumsums vector of new cumsums  (must be initialized!)
 * @return void fill out specified vectors for 1 loyds step
 */
bool FLSPP::single_loyds_step(std::vector<int> &labels, std::vector<Point> &new_centroids, std::vector<double> &new_distances, std::vector<int> &new_labels,
							  std::vector<double> &new_second_closest_distances, std::vector<int> &new_second_closest_labels, std::vector<double> &new_cumsums)
{
	// compute new centroids
	compute_centroids(labels, new_centroids);

	// compute new labels/distances/ cumsum
	bool change = compute_labels_from_given_centroids(new_centroids, new_distances, new_labels, new_second_closest_distances, new_second_closest_labels, new_cumsums);

	return change;
}

/** @brief single loyds step
 *
 * compute new entries after 1 loyds step. Here we also specify some labeling and output the new cenroids/labels etc.
 * @param[in]	labels	(closest) labels of all points in current assignment
 * @param[in]	new_centroids	vector of new centers after 1 loyd step (centroid computation of labels, must be initialized!)
 * @param[in]	new_distances  vector of new distances after loyds step (must be initialized!)
 * @param[in]	new_labels  vector of labels of points to closest center  (must be initialized!)
 * @param[in]	new_cumsums vector of new cumsums  (must be initialized!)
 * @param[in]  new_clustercosts  vector of costs for each center 1 - k(must be initialized!)
 * @return void fill out specified vectors for 1 loyds step
 */
bool FLSPP::single_loyds_step(std::vector<int> &labels, std::vector<Point> &new_centroids, std::vector<double> &new_distances, std::vector<int> &new_labels, std::vector<double> &new_second_closest_distances, std::vector<int> &new_second_closest_labels, std::vector<double> &new_cumsums, std::vector<double> &new_clustercosts)
{
	// compute new centroids
	compute_centroids(labels, new_centroids);

	// compute new labels/distances/ cumsum
	bool change = compute_labels_from_given_centroids(new_centroids, new_distances, new_labels, new_second_closest_distances, new_second_closest_labels, new_cumsums, new_clustercosts);

	return change;
}

///** @brief single loyds step
// *
// * compute new entries after 1 loyds step from current set of labels
// * @return void fill out specified vectors for 1 loyds step
// */
// bool FLSPP::single_loyds_step()
//{
//	update_centroids();
//	compute_centroids(labels, centers);
//
//	bool change compute_labels_from_given_centroids
//}

void FLSPP::set_clustercosts()
{
	clustercosts.resize(centers.size());

	for (std::size_t i = 0; i < points.size(); i++)
	{
		clustercosts[labels[i]] += closest_center_distances[i];
	}
}

void FLSPP_configuration::set_decreasing_clustercosts_order(std::vector<double> &clustgercosts)
{
	// create pairs
	std::vector<std::pair<double, int>> pairs;
	pairs.reserve(clustgercosts.size());
	for (std::size_t i = 0; i < clustgercosts.size(); i++)
	{
		pairs.push_back(std::make_pair(clustgercosts[i], i));
	}
	std::sort(pairs.begin(), pairs.end());
	for (std::size_t i = 0; i < clustgercosts.size(); i++)
	{
		exchange_center_order[i] = pairs[i].second;
	}
}
