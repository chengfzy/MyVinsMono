#include "feature_manager.h"

int FeaturePerId::endFrame() { return start_frame + feature_per_frame.size() - 1; }

FeatureManager::FeatureManager(Matrix3d _Rs[]) : Rs(_Rs) {
    for (int i = 0; i < NUM_OF_CAM; i++) ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[]) {
    for (int i = 0; i < NUM_OF_CAM; i++) {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState() { feature.clear(); }

int FeatureManager::getFeatureCount() {
    int cnt = 0;
    for (auto& it : feature) {
        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2) {
            cnt++;
        }
    }
    return cnt;
}

/**
 * @brief 将当前帧特征加入到feature list中, 并计算每个特征点在次新帧(倒数第二帧)与次次新帧(倒数第三帧)之间的视差,
 * 并返回是否是关键帧. 也就是说当前帧存入到feature list中后, 并不会立即判断当前帧是否是关键帧,
 * 而是判断当前帧的前一帧(即倒数第二帧)
 *
 * @param frame_count   滑窗内帧的个数
 * @param image         某帧所有的特征点信息, <featureId, (cameraId, [x, y, z, u, v, vx, vy)>
 * @param td            IMU和相机的时间差
 * @return true:次新帧是关键帧; false: 不是关键帧
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count,
                                             const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& image,
                                             double td) {
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;  // 倒数第二帧与倒数第三帧之间被跟踪到的特征点的总视差
    int parallax_num = 0;     // 倒数第二帧与倒数第三帧之间被跟踪到的特征点的数量
    last_track_num = 0;       // 倒数第二帧的特征在当前帧中被跟踪的个数

    // 将image中所有的特征点(feature)放入到feature list中
    for (auto& id_pts : image) {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        // 查找feature list是否有该feature
        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(),
                          [feature_id](const FeaturePerId& it) { return it.feature_id == feature_id; });

        if (it == feature.end()) {  // 没有, 则新建一个, 并添加该图像帧
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        } else if (it->feature_id == feature_id) {  // 有的话, 则将该图像帧添加进去
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;  // 上帧的点在该帧中被跟踪的个数+1
        }
    }

    // 图像帧太少(0或1), 或被跟踪的点太少, 则上帧应设为关键帧
    if (frame_count < 2 || last_track_num < 20) {
        return true;
    }

    // 计算倒数第二帧与倒数第三帧之间的平均视差
    for (auto& it_per_id : feature) {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0) {  // 没有跟踪到任何点, 则上帧应设为关键帧
        return true;
    } else {
        // 计算平均视差, 并判断是否是关键帧.
        // 由于这里计算视差是用的规一化坐标, 因此MIN_PARALLAX是视差阈值(像素值)/焦距, VINS中的焦距为固定的虚拟焦距
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow() {
    ROS_DEBUG("debug show");
    for (auto& it : feature) {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto& j : it.feature_per_frame) {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ", j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r) {
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto& it : feature) {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;

            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd& x) {
    int feature_index = -1;
    for (auto& it_per_id : feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        // ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame,
        // it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0) {
            it_per_id.solve_flag = 2;
        } else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures() {
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;
        if (it->solve_flag == 2) feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd& x) {
    int feature_index = -1;
    for (auto& it_per_id : feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector() {
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto& it_per_id : feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

/**
 * @brief 使用SVD对每一个特征点进行三角化求解深度
 *
 * @param Ps    第i帧在世界坐标系下的位置
 * @param tic
 * @param ric
 * @return
 */
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]) {
    for (auto& it_per_id : feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();  // 被观测的帧数
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) {
            continue;
        }

        // 该帧已经被初始化过
        if (it_per_id.estimated_depth > 0) {
            continue;
        }
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto& it_per_frame : it_per_id.feature_per_frame) {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            // R, t为第j帧到第i帧的变换, P为i到j的变换
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            // P = [P1 P2 P3]^T
            // AX=0      A = [A(2*i) A(2*i+1) A(2*i+2) A(2*i+3) ...]^T
            // A(2*i)   = x(i) * P3 - z(i) * P1
            // A(2*i+1) = y(i) * P3 - z(i) * P2
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            // 没什么用
            if (imu_i == imu_j) {
                continue;
            }
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        // it_per_id->estimated_depth = -b / A;
        // it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        // it_per_id->estimated_depth = INIT_DEPTH;

        // 如果估计出来的深度小于0.1, 则替换为初始值5. 好像不太合理呢
        if (it_per_id.estimated_depth < 0.1) {
            it_per_id.estimated_depth = INIT_DEPTH;
        }
    }
}

void FeatureManager::removeOutlier() {
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true) {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R,
                                          Eigen::Vector3d new_P) {
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2) {
                feature.erase(it);
                continue;
            } else {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack() {
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0) feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count) {
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
        it_next++;

        if (it->start_frame == frame_count) {
            it->start_frame--;
        } else {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1) continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0) feature.erase(it);
        }
    }
}

/**
 * @brief 计算某特征点it_per_id在倒数第二帧与倒数第三帧之间的视差.
 *
 * 简单理解, 就是特征点在两帧间被跟踪到了, 计算移动像素差, 注意是在规一化坐标下的像素差
 *
 * @param it_per_id     该特征点
 * @param frame_count   窗口中的帧数
 * @return  视差值
 */
double FeatureManager::compensatedParallax2(const FeaturePerId& it_per_id, int frame_count) {
    // check the second last frame is keyframe or not
    // parallax between seconde last frame and third last frame
    // 计算该特征点在倒数第二帧和倒数第三帧之间的视差
    const FeaturePerFrame& frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame& frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    // 因为特征点都是归一化之后的点, 所以深度均为1, 所以没有去除深度, 下边去除深度, 效果一样
    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    // int r_i = frame_count - 2;
    // int r_j = frame_count - 1;
    // p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    // 视差距离计算, min()中间两个值一样的, 而且与0比较也没意义...
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}