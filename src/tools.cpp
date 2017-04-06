#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  unsigned int n_est = estimations.size();
  if ((n_est == 0) || (n_est != ground_truth.size())) {
    cout << "Zero estimations or estimations and ground truth data of different sizes" << endl;
    return rmse;
  }
  
  for (int i=0; i<n_est; i++) {
    VectorXd difference = estimations[i] - ground_truth[i];
    VectorXd quad_dif = difference.array() * difference.array();
    rmse += quad_dif;
  }
  rmse = rmse / n_est;
  rmse = rmse.array().sqrt();
  
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd J(3, 4);
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  float ss = px * px + py * py;
  float rss = sqrt(ss);
  float rssp3 = ss * rss;

  if (ss == 0.0) {
    ss = 0.00001;
  }
  
  J << px/rss, py/rss, 0, 0,
       -(py/ss), px/ss, 0, 0,
       py*(vx*py-vy*px)/rssp3, px*(vy*px-vx*py)/rssp3, px/rss, py/rss;
  
  return J;
}
