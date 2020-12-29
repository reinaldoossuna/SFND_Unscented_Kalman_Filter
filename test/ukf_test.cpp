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
