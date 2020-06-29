/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  num_particles = 100;  // TODO: Set the number of particles
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  for (int i = 0; i < num_particles; i++) {
    double x0 = particles[i].x;
    double y0 = particles[i].y;
    double theta0 = particles[i].theta;
    if (fabs(yaw_rate) > 0.0001) {
      double theta0_prime = theta0 + yaw_rate*delta_t;
      double ratio_velocity_yaw_rate = velocity / yaw_rate;
      particles[i].x = x0 + ratio_velocity_yaw_rate*(std::sin(theta0_prime) - std::sin(theta0)) + dist_x(gen);
      particles[i].y = y0 + ratio_velocity_yaw_rate*(std::cos(theta0) - std::cos(theta0_prime)) + dist_y(gen);
      particles[i].theta = theta0_prime + dist_theta(gen);
    }
    else {
      particles[i].x = x0 + velocity*delta_t * std::cos(theta0) + dist_x(gen);
      particles[i].y = y0 + velocity*delta_t * std::sin(theta0) + dist_y(gen);
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  size_t observations_size = observations.size();
  size_t predicted_size = predicted.size();
  for (size_t i = 0; i < observations_size; i++) {
    LandmarkObs obs = observations[i];
    double min_distance = std::numeric_limits<double>::max();
    for (size_t j = 0; j < predicted_size; j++) {
      LandmarkObs pred = predicted[j];
      double distance = dist(obs.x, obs.y, pred.x, pred.y);
      if (distance < min_distance) {
        min_distance = distance;
        observations[i].id = pred.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[i];
    vector<LandmarkObs> landmarks_in_range;
    size_t landmarks_list_size = map_landmarks.landmark_list.size();
    for (size_t j = 0; j < landmarks_list_size; j++) {
      Map::single_landmark_s map_landmark = map_landmarks.landmark_list[j];
      double distance = dist(p.x, p.y, map_landmark.x_f, map_landmark.y_f);
      if (distance <= sensor_range) {
        landmarks_in_range.push_back(LandmarkObs{map_landmark.id_i, map_landmark.x_f, map_landmark.y_f});
      }
    }
    
    vector<LandmarkObs> transformed_observations;
    size_t observations_size = observations.size();
    for (unsigned int j = 0; j < observations_size; j++) {
      LandmarkObs obs = observations[j];
      double x = std::cos(p.theta) * obs.x - std::sin(p.theta) * obs.y + p.x;
      double y = std::sin(p.theta) * obs.x + std::cos(p.theta) * obs.y + p.y;
      transformed_observations.push_back(LandmarkObs{obs.id, x, y});
    }

    dataAssociation(landmarks_in_range, transformed_observations);

    particles[i].weight = 1.0;
    size_t transformed_observations_size = transformed_observations.size();
    for (size_t j = 0; j < transformed_observations_size; j++) {
      LandmarkObs transformed_obs = transformed_observations[j];
      int k = 0;
      while (landmarks_in_range[k].id != transformed_obs.id) {
        k++;
      }
      LandmarkObs landmark_in_range = landmarks_in_range[k];
      double exponential_x = std::pow(transformed_obs.x - landmark_in_range.x, 2)/(2*std::pow(std_landmark[0], 2));
      double exponential_y = std::pow(transformed_obs.y - landmark_in_range.y, 2)/(2*std::pow(std_landmark[1], 2));
      double weight = std::exp(-(exponential_x + exponential_y))/(2*M_PI*std_landmark[0]*std_landmark[1]);
      particles[i].weight *= weight;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  weights.clear();
  double max_weight = -1.0;
  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[i];
    if (p.weight > max_weight) {
      max_weight = p.weight;
    }
    weights.push_back(p.weight);
  }

  std::default_random_engine gen;
  std::uniform_int_distribution<int> dist_idx(0, num_particles-1);
  std::uniform_real_distribution<double> dist_beta(0.0, 2*max_weight);

  int idx = dist_idx(gen);
  double beta = 0.0;
  vector<Particle> new_particles;
  for (int i = 0; i < num_particles; i++) {
    beta += dist_beta(gen);
    while (beta > weights[idx]) {
      beta -= weights[idx];
      idx = (idx + 1) % num_particles;
    }
    new_particles.push_back(particles[idx]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}