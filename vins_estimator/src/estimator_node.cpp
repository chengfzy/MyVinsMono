#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <stdio.h>
#include <condition_variable>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;             // 接受到的IMU数据
queue<sensor_msgs::PointCloudConstPtr> feature_buf;  // 接受到的特征数据
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;        // 上一帧IMU时间戳
Eigen::Vector3d tmp_P;     // 上一帧的位置
Eigen::Quaterniond tmp_Q;  // 上一帧的旋转
Eigen::Vector3d tmp_V;     // 上一帧的速度
Eigen::Vector3d tmp_Ba;    // 上一帧的ba
Eigen::Vector3d tmp_Bg;    // 上一帧的bg
Eigen::Vector3d acc_0;  // 上一帧的加速度测量值, 可能是IMU的测量值, 也可能是通过图像时间戳进行插值后的值
Eigen::Vector3d gyr_0;  // 上一帧的陀螺仪测量值, 可能是IMU的测量值, 也可能是通过图像时间戳进行插值后的值
bool init_feature = 0;  // 图像特征是否初始化, 主要用来判断是否是第一次接收到特征数据
bool init_imu = 1;      // IMU是否初始化, 主要有来判断是否是第一次接收到IMU数据
double last_imu_t = 0;  // 上一帧IMU的时间戳

/**
 * @brief 从IMU测量值imu_msg, 使用中值积分对IMU进行预测, 主要是预测[p, Q, v]
 *
 * @param imu_msg   接收到的IMU数据
 */
void predict(const sensor_msgs::ImuConstPtr& imu_msg) {
    double t = imu_msg->header.stamp.toSec();
    if (init_imu) {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * @brief 从estimator得到滑动窗口当前图像帧的状态(p, Q, v, bg, ba, a, g), 对imu_buf中的剩余的IMU数据进行递推,
 * 并预测当前的状态量(p, Q, v, a, g)
 */
void update() {
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    // 取出滑窗内的acc和gyro, 主要是滑窗可能对IMU数据进行了插值
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop()) {
        predict(tmp_imu_buf.front());
    }
}

/**
 * @brief 将IMU和图像数据进行对齐打包, 即一帧图像对应多个IMU值
 *
 * @return {(IMUs, img)}
 */
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> getMeasurements() {
    // 测量数据, 一帧图像对应多个IMU
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true) {
        if (imu_buf.empty() || feature_buf.empty()) {
            return measurements;
        }

        // 等待IMU: IMU最后一个数据的时间戳要大于第一个图像特征的时间戳
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td)) {
            // ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        // IMU每一个数据的时间戳要小于第一个图像特征的时间戳
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td)) {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        // 将某帧图像, 及该帧对应的IMU值添加到测量值中
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td) {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        // 这里把下一个IMU也放进去了, 但没有pop, 因此当前图像帧和下一个图像帧或共用这个IMU
        IMUs.emplace_back(imu_buf.front());

        if (IMUs.empty()) {
            ROS_WARN("no imu between two image");
        }
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

/**
 * @brief IMU回调函数, 使用接受的IMU数据进行预测, 即状态递推, 并更新显示
 *
 * @param imu_msg
 * @return
 */
void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    // 保证IMU数据时间戳是一直增加的
    if (imu_msg->header.stamp.toSec() <= last_imu_t) {
        ROS_WARN("imu message in disorder!");
        return;
    }

    // 更新一帧的IMU时间戳
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();  // 更新两次?没啥意义呢

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);  // IMU递推
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
            // 发布最新的由IMU递推得到的[p, Q, v, 时间戳]
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
        }
    }
}

/**
 * @brief feature回调函数, 将feature_msg加入到feature_buf
 *
 * @param feature_msg
 * @return
 */
void feature_callback(const sensor_msgs::PointCloudConstPtr& feature_msg) {
    if (!init_feature) {
        // skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

// restart回调函数, 清空feature_buf和imu_buf, 重置estimator和时间
void restart_callback(const std_msgs::BoolConstPtr& restart_msg) {
    if (restart_msg->data == true) {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while (!feature_buf.empty()) {
            feature_buf.pop();
        }
        while (!imu_buf.empty()) {
            imu_buf.pop();
        }
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

// relocalization回调函数, 将points_msg放入relo_buf
void relocalization_callback(const sensor_msgs::PointCloudConstPtr& points_msg) {
    // printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
/**
 * @brief VIO主线程
 *
 * 1. 等待并获取measurements, {(IMUs, img)}
 * 2. estimator.processIMU()进行IMU预积分
 * 3. estimator.setReloFrame()设置重定位帧
 * 4. estimator.processImage()处理图像帧: 初始化, 紧耦合优化
 * @return
 */
void process() {
    while (true) {
        // 等待IMU和图像数据
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&] { return (measurements = getMeasurements()).size() != 0; });
        lk.unlock();

        m_estimator.lock();
        for (auto& measurement : measurements) {
            // 使用IMU数据进行预积分
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto& imu_msg : measurement.first) {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                if (t <= img_t) {
                    if (current_time < 0) current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    // printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                } else {
                    // 如果该IMU数据属于该图像, 但时间戳介于当前图像与下一图像之间, 则对其插值进行处理
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    // printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            // 取出最后一个重定位帧
            while (!relo_buf.empty()) {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL) {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++) {
                    Vector3d u_v_id;  // u, v, featureId
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1],
                                relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4],
                                   relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            // 处理图像数据
            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());
            TicToc t_s;
            // 每个特征点的map, <featureId, (cameraId, [x, y, z, u, v, vx, vy)>
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++) {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;  // 归一化3D点, z=1
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];  // 像素坐标, (u, v)
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];  // 像素速度, (vx, vy)
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }
            // 处理图像特征
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            // 给RVIZ发送topic
            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL) {
                pubRelocalization(estimator);
            }
            // ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
            update();
        }
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
