#ifndef MEASUREMENT_PACKAGE_H_
#define MEASUREMENT_PACKAGE_H_

#include "Eigen/Dense"

using Eigen::VectorXd;


class MeasurementPackage {
    public:
        enum SensorType { LASER, RADAR } sensor_type_;
        long long timestamp_;
        VectorXd raw_measurements_;
        VectorXd ground_truth_;
};


#endif