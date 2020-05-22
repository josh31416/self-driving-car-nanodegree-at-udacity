#include "extended_kalman_filter.h"
#include <math.h>
#include <iostream>

using std::cout;
using std::endl;

ExtendedKalmanFilter::ExtendedKalmanFilter() {
    x_ = VectorXd(4);
    P_ = MatrixXd(4, 4);
    F_ = MatrixXd(4, 4);
    Q_ = MatrixXd(4, 4);
}


ExtendedKalmanFilter::~ExtendedKalmanFilter() {}


void ExtendedKalmanFilter::Predict() {
    // prior state
    x_ = F_ * x_; // + 0
    // prior state covariance
    P_ = F_ * P_ * F_.transpose() + Q_;
}


void ExtendedKalmanFilter::Update(const VectorXd &z) {
    // reuse op
    MatrixXd Ht = H_.transpose();

    // innovation
    VectorXd y = z - H_ * x_;
    // innovation covariance
    MatrixXd S = H_ * P_ * Ht + R_;
    // optimal kalman gain
    MatrixXd K = P_ * Ht * S.inverse();

    // posterior state
    x_ = x_ + (K * y);
    // posterior state covariance
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}


void ExtendedKalmanFilter::UpdateEKF(const VectorXd &z) {
    // reuse op
    MatrixXd Ht = H_.transpose();

    // innovation
    VectorXd y = z - h(x_);
    // theta to range [-pi, pi]
    while(y(1) > M_PI){
      y(1) -= 2*M_PI;
    }
    while(y(1) < -M_PI){
      y(1) += 2*M_PI;
    }
    // innovation covariance
    MatrixXd S = H_ * P_ * Ht + R_;
    // optimal kalman gain
    MatrixXd K = P_ * Ht * S.inverse();

    // posterior state
    x_ = x_ + (K * y);
    // posterior state covariance
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}


void ExtendedKalmanFilter::calculateHj() {
    H_ = calculateJacobian(x_);
}


VectorXd h(const VectorXd &state) {
    VectorXd result(3);
    // extract state
    float x = state(0);
    float y = state(1);
    float vx = state(2);
    float vy = state(3);
    // ro
    result(0) = sqrt(x*x + y*y);
    // theta
    float theta = atan2(y, x);
    result(1) = theta;
    // ro_dot
    result(2) = (x*vx + y*vy)/result(0);

    return result;
}


MatrixXd calculateJacobian(const VectorXd &state) {
    MatrixXd Hj(3, 4);
    // extract state
    float x = state(0);
    float y = state(1);
    float vx = state(2);
    float vy = state(3);
    // reuse ops
    float c1 = x*x + y*y;
    float c2 = sqrt(c1);
    float c3 = (c1 * c2);

    // check division by zero
    if (fabs(c1) < 0.0001) {
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
        return Hj;
    }

    // compute the Jacobian matrix
    Hj << (x/c2),             (y/c2),             0,    0,
          -(y/c1),            (x/c1),             0,    0,
          y*(y*vx - x*vy)/c3, x*(x*vy - y*vx)/c3, x/c2, y/c2;

    return Hj;
}