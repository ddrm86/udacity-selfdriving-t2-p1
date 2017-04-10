#include "kalman_filter.h"
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

VectorXd KalmanFilter::CalculateZpred(const VectorXd &z) {
  VectorXd z_pred;
  if (z.size() == 2) {
    z_pred = H_ * x_;
  } else {
    double px = x_(0);
    double py = x_(1);
    double vx = x_(2);
    double vy = x_(3);
  
    double rho = sqrt(px*px + py*py);
    double phi = atan2(py, px);
    double rho_dot = (px*vx + py*vy)/rho;
    z_pred = VectorXd(3);
    z_pred << rho, phi, rho_dot;
  }
  return z_pred;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd z_pred = CalculateZpred(z);
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  
  x_ = x_ + (K * y);
  unsigned long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;  
}

