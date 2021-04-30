#pragma once
#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include "../factor/imu_factor.h"
#include "../feature_manager.h"
#include "../utility/utility.h"

using namespace Eigen;
using namespace std;

/**
 * @brief 图像帧
 *
 * @return
 */
class ImageFrame {
  public:
    ImageFrame(){};
    ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t)
        : t{_t}, is_key_frame{false} {
        points = _points;
    };

    // 该帧所有的特征点信息, <featureId, (cameraId, [x, y, z, u, v, vx, vy)>
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> points;
    double t;                          // 时间戳
    Matrix3d R;                        // R
    Vector3d T;                        // T
    IntegrationBase* pre_integration;  // 预积分
    bool is_key_frame;                 // 是否是关键帧
};

bool VisualIMUAlignment(map<double, ImageFrame>& all_image_frame, Vector3d* Bgs, Vector3d& g, VectorXd& x);