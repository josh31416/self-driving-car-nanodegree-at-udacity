#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "Eigen/Dense"
#include "measurement_package.h"
#include "extended_kalman_filter.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;


class FusionEKF {
    public:
        /**
         *  Constructor
         */
        FusionEKF();

        /**
         *  Destructor
         */
        virtual ~FusionEKF();

        /**
         * Run the whole flow of the Kalman Filter from here.
         */
        void ProcessMeasurement(const MeasurementPackage &measurement_package);

        // Extended Kalman Filter
        ExtendedKalmanFilter ekf_;
    private:
        // is first measurement
        bool is_initialized_;

        //previous timestamp
        long long previous_timestamp_;

        // covariance matrices
        MatrixXd R_laser_;
        MatrixXd R_radar_;
        // observational model
        MatrixXd H_laser_;

        // noise covariances for Q
        float noise_ax_;
        float noise_ay_;
};


#endif