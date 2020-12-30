#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <iostream>

#include "ukf.h"
#include "measurement_package.h"

const double err = 0.001;

void TEST_MATRIX(Eigen::MatrixXd& A, Eigen::MatrixXd& B) {
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            EXPECT_NEAR(A(i, j), B(i, j), err) << "i = " << i << " j = " << j;
        }
    }
}
void TEST_VECTOR(Eigen::VectorXd& a, Eigen::VectorXd& b) {
    for (int i = 0; i < a.size(); i++) {
            EXPECT_NEAR(a(i), b(i), err) << "i = " << i;
    }
}
struct ukf_test : testing::Test {
    std::unique_ptr<UKF> ukf;

    ukf_test() {
        ukf = std::make_unique<UKF>();

        // add fake data
        // base in past exercicies
        ukf->std_a_ = 0.2;
        ukf->std_yawdd_ = 0.2;
        ukf->x_ <<
            5.7441,
            1.3800,
            2.2049,
            0.5015,
            0.3528;


        ukf->P_ <<
            0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
            -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
            0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
            -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
            -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

    }
};
TEST_F(ukf_test, constructor) {
    //dump test
    Eigen::VectorXd x_exp = Eigen::VectorXd(5);
    x_exp <<
        5.7441,
        1.3800,
        2.2049,
        0.5015,
        0.3528;
    Eigen::MatrixXd P_exp = Eigen::MatrixXd(5, 5);
    P_exp <<
        0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
        -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
        0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
        -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
        -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

    EXPECT_DOUBLE_EQ(0.2, ukf->std_a_);
    EXPECT_DOUBLE_EQ(0.2, ukf->std_yawdd_);
    TEST_VECTOR(x_exp, ukf->x_);
    TEST_MATRIX(P_exp, ukf->P_);
}
TEST_F(ukf_test, gen_sigma_points) {
    Eigen::MatrixXd Xsig_arg = ukf->generate_sigma_points();

    Eigen::MatrixXd Xsig_exp = Eigen::MatrixXd(7, 15);
    Xsig_exp <<
        5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
        1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
        2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
        0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
        0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
            0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641,        0,
            0,        0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641;

    TEST_MATRIX(Xsig_exp, Xsig_arg);
}
TEST_F(ukf_test, predict_sigma) {
    double delta_t = 0.1;
    Eigen::MatrixXd Xsig_arg = ukf->generate_sigma_points();
    ukf->predict_sigma_points(Xsig_arg, delta_t);

    Eigen::MatrixXd Xsig_pred_exp = Eigen::MatrixXd(5, 15);
    Xsig_pred_exp <<
    5.93553,  6.06251,  5.92217,   5.9415,  5.92361,  5.93516,  5.93705,  5.93553,  5.80832,  5.94481,  5.92935,  5.94553,  5.93589,  5.93401,  5.93553,
    1.48939,  1.44673,  1.66484,  1.49719,    1.508,  1.49001,  1.49022,  1.48939,   1.5308,  1.31287,  1.48182,  1.46967,  1.48876,  1.48855,  1.48939,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,  2.23954,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,  2.17026,   2.2049,
    0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372,  0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188,  0.53678, 0.535048,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528, 0.387441, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528, 0.318159;

    TEST_MATRIX(Xsig_pred_exp, ukf->Xsig_pred_);
}
TEST_F(ukf_test, predict_mean_cov) {
    double delta_t = 0.1;
    Eigen::MatrixXd Xsig_arg = ukf->generate_sigma_points();
    ukf->predict_sigma_points(Xsig_arg, delta_t);
    ukf->predict_mean_covariance();

    Eigen::VectorXd x_exp = Eigen::VectorXd(5);
    x_exp <<
        5.93445,
        1.48885,
        2.2049,
        0.53678,
        0.3528;

    Eigen::MatrixXd P_exp = Eigen::MatrixXd(5, 5);
    P_exp <<
        0.0054808, -0.00249899,  0.00340521,  -0.0035741, -0.00309082,
        -0.00249899,   0.0110551,  0.00151803,  0.00990779,  0.00806653,
        0.00340521,  0.00151803,   0.0057998, 0.000780142, 0.000800107,
        -0.0035741,  0.00990779, 0.000780142,   0.0119239,     0.01125,
        -0.00309082,  0.00806653, 0.000800107,     0.01125,      0.0127;

    TEST_VECTOR(x_exp, ukf->x_);
    TEST_MATRIX(P_exp, ukf->P_);
}
TEST_F(ukf_test, predict_measurement_radar) {
    double delta_t = 0.1;
    Eigen::MatrixXd Xsig_arg = ukf->generate_sigma_points();
    ukf->predict_sigma_points(Xsig_arg, delta_t);
    ukf->predict_mean_covariance();
    Eigen::VectorXd z_pred;
    Eigen::MatrixXd S;
    ukf->predict_measurement_radar(&z_pred, &S);

    Eigen::VectorXd z_exp = Eigen::VectorXd(3);
    z_exp <<
        6.11934,
        0.245833,
        2.10274;

    Eigen::MatrixXd S_exp = Eigen::MatrixXd(3, 3);
    S_exp <<
        0.0946306, -0.000145108,   0.00408754,
        -0.000145108,   0.00121798, -0.000781354,
        0.00408754, -0.000781354,    0.0980469;

    TEST_VECTOR(z_exp, z_pred);
    TEST_MATRIX(S_exp, S);
}

TEST_F(ukf_test, update_radar) {
    double delta_t = 0.1;
    Eigen::MatrixXd Xsig_arg = ukf->generate_sigma_points();
    ukf->predict_sigma_points(Xsig_arg, delta_t);
    ukf->predict_mean_covariance();
    Eigen::VectorXd z_pred;
    Eigen::MatrixXd S;
    ukf->predict_measurement_radar(&z_pred, &S);

    MeasurementPackage mp;
    mp.sensor_type_ = MeasurementPackage::SensorType::RADAR;
    mp.raw_measurements_ = Eigen::VectorXd(3);
    mp.raw_measurements_ <<
                            5.9214,   // rho in m
                            0.2187,   // phi in rad
                            2.0062;   // rho_dot in m/s

    ukf->UpdateRadar(mp);

    Eigen::VectorXd x_exp = Eigen::VectorXd(5);
    x_exp <<
        5.93337,
        1.44926,
        2.18927,
        0.505393,
        0.328322;

    Eigen::MatrixXd P_exp = Eigen::MatrixXd(5, 5);
    P_exp <<
        0.00473199, -0.00147372,  0.00304273, -0.00245549, -0.00213064,
        -0.00147372,  0.00817367,  0.00146719,  0.00719161,  0.00582479,
        0.00304273,  0.00146719,  0.00538703, 0.000901972, 0.000946797,
        -0.00245549,  0.00719161, 0.000901972,  0.00929299,  0.00905869,
        -0.00213064,  0.00582479, 0.000946797,  0.00905869,   0.0108689;

    TEST_VECTOR(x_exp, ukf->x_);
    TEST_MATRIX(P_exp, ukf->P_);
}
TEST_F(ukf_test, predict_measurement_lidar) {
    double delta_t = 0.1;
    Eigen::MatrixXd Xsig_arg = ukf->generate_sigma_points();
    ukf->predict_sigma_points(Xsig_arg, delta_t);
    ukf->predict_mean_covariance();
    Eigen::VectorXd z_pred;
    Eigen::MatrixXd S;
    ukf->predict_measurement_lidar(&z_pred, &S);

    Eigen::VectorXd z_exp = Eigen::VectorXd(2);
    z_exp <<
        5.93445,
        1.48885;
    Eigen::MatrixXd S_exp = Eigen::MatrixXd(2, 2);
    S_exp <<
        0.0279808, -0.00249899,
        -0.00249899,   0.0335551;

    TEST_VECTOR(z_exp, z_pred);
    TEST_MATRIX(S_exp, S);
}
