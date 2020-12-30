#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = .3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = .3;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // set state dimension
  n_x_ = 5;

  // set augmented dimension
  n_aug_ = 7;

  // set spreading parameter
  lambda_ = 3 - n_aug_;

  // initialize sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // set vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  double weights0 = lambda_ / (lambda_ + n_aug_);
  double weigth = 0.5 / (lambda_ + n_aug_);
  weights_(0) = weights0;
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
    weights_(i) = weigth;
  }

  // measurement dimension of radar
  // r, phi, r_dot
  n_z_radar = 3;

  // set measerement noise covariance matrix
  R_radar_= MatrixXd(n_z_radar, n_z_radar);
  R_radar_ <<
    std_radr_ * std_radr_, 0, 0,
    0, std_radphi_ * std_radphi_,0,
    0, 0, std_radrd_ * std_radrd_;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];

      double px = rho * cos(phi);
      double py = rho * sin(phi);

      double vx = rho_dot * cos(phi);
      double vy = rho_dot * sin(phi);
      double v = sqrt(vx * vx + vy * vy);

      x_ <<
        px,
        py,
        v,
        0,
        0;

      P_ = MatrixXd::Identity(5, 5);

    } else {
      // set lidar initialization
      return;
    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }
  auto dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR
      && use_radar_) {
    UpdateRadar(meas_package);
  } else {
    // set updatelidar
  }
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

MatrixXd UKF::generate_sigma_points() {
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented mean state, last two values are set to 0
  x_aug.head(x_.size()) = x_;
  x_aug(n_aug_ - 2) = 0;
  x_aug(n_aug_ - 1) = 0;

  // Create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_aug_ - 2, n_aug_ - 2) = std_a_ * std_a_;
  P_aug(n_aug_ - 1, n_aug_ - 1) = std_yawdd_ * std_yawdd_;

  //calculate square root of P
  MatrixXd L = P_aug.llt().matrixL();

  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  return Xsig_aug;
}

void UKF::predict_sigma_points(MatrixXd& Xsig_aug, double dt) {
  for (int i = 0; i < Xsig_aug.cols(); i++) {
    auto px       = Xsig_aug(0, i);
    auto py       = Xsig_aug(1, i);
    auto v        = Xsig_aug(2, i);
    auto yaw      = Xsig_aug(3, i);
    auto yawd     = Xsig_aug(4, i);
    auto nu_a     = Xsig_aug(5, i);
    auto nu_yawwd = Xsig_aug(6, i);

    // predicted states
    double px_p, py_p;


    // avoid division by 0
    if (fabs(yawd) > 0.001) {
      px_p = px + v/yawd * (sin(yaw + yawd * dt) - sin(yaw));
      py_p = py + v/yawd * (cos(yaw) - cos(yaw + yawd * dt));
    } else {
      px_p = px + v * dt * cos(yaw);
      py_p = py + v * dt * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * dt;
    double yawd_p = yawd;

    // add noise
    px_p   += 0.5 * dt * dt * nu_a * cos(yaw);
    py_p   += 0.5 * dt * dt * nu_a * sin(yaw);
    v_p    += dt * nu_a;
    yaw_p  +=  0.5 * dt * dt * nu_yawwd;
    yawd_p += dt * nu_yawwd;

    Xsig_pred_.col(i).fill(0);
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}

void UKF::normalize_angle(double& a){
    while(a >  M_PI) a -= 2. * M_PI;
    while(a < -M_PI) a += 2. * M_PI;
}

VectorXd UKF::weigthed_mean(MatrixXd& X) {
  VectorXd y_out = VectorXd(X.rows());
  y_out.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    y_out += weights_(i) * X.col(i);
  }
  return y_out;
}

MatrixXd UKF::weigthed_covariance(MatrixXd& X, VectorXd& mean_x, int index_normalize) {
  MatrixXd Y = MatrixXd(X.rows(),X.rows());
  Y.fill(0.0);
  for (int i = 0; i < X.cols(); i++) {
    VectorXd diff = X.col(i) - mean_x;

    //  yaw need to be -PI < yaw < PI;
    normalize_angle(diff(index_normalize));

    Y += weights_(i) * diff * diff.transpose();
  }
  return Y;
}

void UKF::predict_mean_covariance() {
  x_ = weigthed_mean(Xsig_pred_);
  P_ = weigthed_covariance(Xsig_pred_, x_, 3);
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */
  MatrixXd Xsig_aug = generate_sigma_points();
  predict_sigma_points(Xsig_aug, delta_t);
  predict_mean_covariance();
}

MatrixXd UKF::sigma_2_radar() {
  // r = sqrt(x^2 + y^2)
  // phi = atan(y / x)
  // c = cos(yaw) s = sin(yaw)
  // r_dot = (x * c * v + y * s * v)/sqrt(x^2 + y^2)

  MatrixXd Zsig = MatrixXd(n_z_radar, 2 * n_aug_ + 1);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    // calculate measurement
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
    Zsig(1, i) = atan2(p_y, p_x);
    Zsig(2, i) = (p_x * cos(yaw) * v + p_y * sin(yaw) * v) / Zsig(0, i);
  }

  return Zsig;
}

void UKF::predict_measurement_radar(VectorXd* z_pred_out, MatrixXd* S_out) {
  MatrixXd Zsig = sigma_2_radar();

  // mean predicted measurement
  VectorXd z_pred = weigthed_mean(Zsig);

  // covariance matrix S_
  MatrixXd S = weigthed_covariance(Zsig, z_pred, 1);

  // add measurment noise
  S += R_radar_;

  // return
  *S_out = S;
  *z_pred_out = z_pred;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  MatrixXd Zsig = sigma_2_radar();

  // cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar);
  Tc.fill(0.0);

  VectorXd z_pred;
  MatrixXd S;
  predict_measurement_radar(&z_pred, &S);
  // calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    normalize_angle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    normalize_angle(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain
  MatrixXd K = Tc * S.inverse();

  // get data from sensor
  VectorXd z = meas_package.raw_measurements_;
  // update state mean and covariance
  VectorXd z_diff = z - z_pred;
  normalize_angle(z_diff(1));

  x_ += K * z_diff;
  P_ -= K * S * K.transpose();
}
