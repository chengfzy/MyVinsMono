#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <algorithm>
#include <list>
#include <numeric>
#include <vector>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/assert.h>
#include <ros/console.h>

#include "parameters.h"

/**
 * @brief 每一帧图像中的特征
 *
 * @return
 */
class FeaturePerFrame {
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1>& _point, double td) {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        cur_td = td;
    }
    double cur_td;      // IMU与图像之间的时间差
    Vector3d point;     // 图像的归一化3D点(z=1, 或模长为1)
    Vector2d uv;        // 原始的未去畸图像点
    Vector2d velocity;  // 像素移动速度
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

/**
 * @brief 某Feature ID下所有的Feature
 *
 * @return
 */
class FeaturePerId {
  public:
    const int feature_id;                       // 特征点ID
    int start_frame;                            // 最开始被哪帧观测到
    vector<FeaturePerFrame> feature_per_frame;  // 该特征在第一帧的信息

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag;  // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame), used_num(0), estimated_depth(-1.0), solve_flag(0) {}

    int endFrame();
};

class FeatureManager {
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& image,
                                 double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    // void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd& x);
    void removeFailures();
    void clearDepth(const VectorXd& x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R,
                              Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId& it_per_id, int frame_count);
    const Matrix3d* Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif