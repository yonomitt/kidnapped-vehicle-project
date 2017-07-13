/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cfloat>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // If the particle filter is already initialized, don't do it again
  if (is_initialized) {
    return;
  }

  // Set the number of particles
  num_particles = 100;

  // Setup a random number generator
  default_random_engine gen;

  // Setup a normal distribution for x, y, and theta
  normal_distribution<double> nd_x(x, std[0]);
  normal_distribution<double> nd_y(y, std[1]);
  normal_distribution<double> nd_theta(theta, std[2]);

  // Initialize the particles
  for (int i = 0; i < num_particles; i++) {

    Particle p;

    // Set the id to the particle number
    p.id = i;

    // Generate x, y, and theta based on a normal distribution about their means
    p.x = nd_x(gen);
    p.y = nd_y(gen);
    p.theta = nd_theta(gen);

    // All initial weights are 1.0
    p.weight = 1.0;

    // Add the particle to the vector of particles
    particles.push_back(p);

    // Add the particle's weight to the vector of all particle weights
    weights.push_back(p.weight);
  }

  // Mark the particle filter as initialized
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // Setup a random number generator
  default_random_engine gen;

  // Setup a normal distribution for x, y, and theta
  normal_distribution<double> nd_x(0.0, std_pos[0]);
  normal_distribution<double> nd_y(0.0, std_pos[1]);
  normal_distribution<double> nd_theta(0.0, std_pos[2]);
   
  // Precalculate theta_dot * dt to avoid doing it many times
  double yaw_rate_dt = yaw_rate * delta_t;

  // Precalculate v / theta_dot to avoid doing it many times
  double v_over_yaw_rate = velocity / yaw_rate;

  // Loop through all particles and add the measurment plus some noise to them
  for (int i = 0; i < num_particles; i++) {

    Particle p = particles[i];

    // Update the measurements
    p.x += v_over_yaw_rate * (sin(p.theta + yaw_rate_dt) - sin(p.theta));
    p.y += v_over_yaw_rate * (sin(p.theta + yaw_rate_dt) - sin(p.theta));
    p.theta += yaw_rate_dt;

    // Add some Gaussian noise
    p.x += nd_x(gen);
    p.y += nd_y(gen);
    p.theta += nd_theta(gen);

    // Put the particle back in the vector
    particles[i] = p;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

  // Loop through each observed measurements
  for (int i = 0; i < observations.size(); i++) {

    LandmarkObs o = observations[i];

    // Initialize the minimum distance (squared) to be the maximum double value
    double min_dist_sq = DBL_MAX;

    // Loop through all predicted measurements
    for (LandmarkObs p : predicted) {

      // Calculate the square of the distance between the predicted and observed measurement.
      // We don't need to perform the square root because we only care about relative distances
      // and the square root would cost more time.
      double dx = p.x - o.x;
      double dy = p.y - o.y;
      double dist_sq = (dx * dx) + (dy * dy);

      // If the squared distance is less than the minimum
      if (dist_sq < min_dist_sq) {

        // Record the map id of the predicted measurement
        observations[i].id = p.id;

        // Set the minimum to the current one
        min_dist_sq = dist_sq;
      }
    } 
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
   particle.sense_x = sense_x;
   particle.sense_y = sense_y;

   return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
