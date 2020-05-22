#ifndef ExtendedKalmanFilter_H_
#define ExtendedKalmanFilter_H_

#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;


VectorXd h(const VectorXd &state);
MatrixXd calculateJacobian(const VectorXd &state);

class ExtendedKalmanFilter {
    public:
        /**
         *  Constructor
         */
        ExtendedKalmanFilter();

        /**
         *  Destructor
         */
        virtual ~ExtendedKalmanFilter();

        /**
         * Prediction Predicts the state and the state covariance
         * using the process model
         */
        void Predict();

        /**
         * Updates the state by using standard Kalman Filter equations
         * @param z The measurement at k+1
         */
        void Update(const VectorXd &z);

        /**
         * Updates the state by using Extended Kalman Filter equations
         * @param z The measurement at k+1
         */
        void UpdateEKF(const VectorXd &z);

        /**
         * A helper method to calculate the Jacobian Matrix
         */
        void calculateHj();

        // state vector
        VectorXd x_;

        // state covariance matrix
        MatrixXd P_;

        // state transition matrix
        MatrixXd F_;

        // process covariance matrix
        MatrixXd Q_;

        // observational model
        MatrixXd H_;

        // measurement covariance matrix
        MatrixXd R_;
};


#endif