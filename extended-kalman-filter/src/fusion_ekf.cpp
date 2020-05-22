#include "fusion_ekf.h"

using std::cout;
using std::endl;


FusionEKF::FusionEKF() {
    is_initialized_ = false;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
                0,      0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0,      0,
                0,    0.0009, 0,
                0,    0,      0.09;
    
    // observational model - laser
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;

    // initialize state transition matrix
    ekf_.F_ << 1, 0, 1, 0,
               0, 1, 0, 1,
               0, 0, 1, 0,
               0, 0, 0, 1;

    // initialize covariance matrix
    ekf_.P_ << 1, 0, 0,    0,
               0, 1, 0,    0,
               0, 0, 1000, 0,
               0, 0, 0,    1000;
    
    noise_ax_ = 9;
    noise_ay_ = 9;
}


FusionEKF::~FusionEKF() {}


void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_package) {
    if (!is_initialized_) {
        // initalize state
        if (measurement_package.sensor_type_ == MeasurementPackage::LASER) {
            float x = measurement_package.raw_measurements_[0];
            float y = measurement_package.raw_measurements_[1];
            ekf_.x_ << x, y, 0, 0;
        } else if (measurement_package.sensor_type_ == MeasurementPackage::RADAR) {
            float ro = measurement_package.raw_measurements_[0];
            float theta = measurement_package.raw_measurements_[1];
            ekf_.x_ << ro * cos(theta), ro * sin(theta), 0, 0;
        }

        // initialize timestamp
        previous_timestamp_ = measurement_package.timestamp_;

        // set filter as initialized
        is_initialized_ = true;

        // print init
        // cout << "Init" << endl;
        // cout << "x_ = " << endl << ekf_.x_ << endl;
        // cout << "P_ = " << endl << ekf_.P_ << endl;
        // cout << endl;
    } else {
        // calculate time passed in seconds
        float dt = (measurement_package.timestamp_ - previous_timestamp_) / 1000000.0;
        previous_timestamp_ = measurement_package.timestamp_;

        // integrate time in state transition matrix F
        ekf_.F_(0, 2) = dt;
        ekf_.F_(1, 3) = dt;

        // set process covariance matrix Q
        float dt_2 = dt * dt;
        float dt_3 = dt_2 * dt;
        float dt_4 = dt_3 * dt;

        ekf_.Q_ << dt_4/4*noise_ax_, 0,                dt_3/2*noise_ax_, 0,
                   0,                dt_4/4*noise_ay_, 0,                dt_3/2*noise_ay_,
                   dt_3/2*noise_ax_, 0,                dt_2*noise_ax_,   0,
                   0,                dt_3/2*noise_ay_, 0,                dt_2*noise_ay_;
        
        // predict step
        ekf_.Predict();

        // print prior
        // cout << "Prior" << endl;
        // cout << "x_ = " << endl << ekf_.x_ << endl;
        // cout << "P_ = " << endl << ekf_.P_ << endl;
        // cout << endl;

        // update step
        if (measurement_package.sensor_type_ == MeasurementPackage::LASER) {
            // laser covariance R
            ekf_.R_ = R_laser_;
            // laser observational model H
            ekf_.H_ = H_laser_;
            // update step
            ekf_.Update(measurement_package.raw_measurements_);
        } else if (measurement_package.sensor_type_ == MeasurementPackage::RADAR) {
            // radar covariance R
            ekf_.R_ = R_radar_;
            // set Hj
            ekf_.calculateHj();
            // update step
            ekf_.UpdateEKF(measurement_package.raw_measurements_);
        }

        // print posterior
        // cout << "Posterior" << endl;
        // cout << "x_ = " << endl << ekf_.x_ << endl;
        // cout << "P_ = " << endl << ekf_.P_ << endl;
        // cout << endl;
    }
}