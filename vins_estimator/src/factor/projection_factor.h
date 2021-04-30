#pragma once

#include <ceres/ceres.h>
#include <ros/assert.h>
#include <Eigen/Dense>
#include "../parameters.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"

/**
 * @brief 基于逆深度的重投影误差.
 *
 * 一个3D点同时被两个Frame i,j观测到, 从而形成重投影误差. 这里的第i帧更准确说是3D点首次被观测到的帧.
 *
 * 优化变量有
 *  1. FrameI的位姿
 *  2. FrameJ的位姿
 *  3. IMU-相机之间的位姿
 *  4. 逆深度. 3D点在第i帧的深度
 *
 * @return
 */
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1> {
  public:
    ProjectionFactor(const Eigen::Vector3d& _pts_i, const Eigen::Vector3d& _pts_j);
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;
    void check(double** parameters);

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;  // 统计Evaluate()函数的总共调用时间
};
