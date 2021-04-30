#pragma once

#include <ceres/ceres.h>
#include <pthread.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <cstdlib>
#include <unordered_map>

#include "../utility/tic_toc.h"
#include "../utility/utility.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo {
    ResidualBlockInfo(ceres::CostFunction* _cost_function, ceres::LossFunction* _loss_function,
                      std::vector<double*> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function),
          loss_function(_loss_function),
          parameter_blocks(_parameter_blocks),
          drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction* cost_function;
    ceres::LossFunction* loss_function;
    std::vector<double*> parameter_blocks;  // 优化变量内存地址
    std::vector<int> drop_set;              // 需要被边缘化的变量地址的id,也就是parameter_blocks的id

    double** raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;

    int localSize(int size) { return size == 7 ? 6 : size; }
};

struct ThreadsStruct {
    std::vector<ResidualBlockInfo*> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size;  // global size
    std::unordered_map<long, int> parameter_block_idx;   // local size
};

class MarginalizationInfo {
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo* residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double*> getParameterBlocks(std::unordered_map<long, double*>& addr_shift);

    std::vector<ResidualBlockInfo*> factors;  // 视觉观测项, IMU观测及上一次的边缘化项
    int m, n;                                 // m为要边缘化的变量个数, n为要保留下来的变量个数
    std::unordered_map<long, int> parameter_block_size;  // <优化变量内存地址, localSize>
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx;  // <优化变量内存地址, 在矩阵(parameter_block_size)中的id>
    std::unordered_map<long, double*> parameter_block_data;  // <优化变量内存地址, 数据>

    std::vector<int> keep_block_size;      // 按顺序存放parameter_block_size中被保留的优化变量
    std::vector<int> keep_block_idx;       // 按顺序存放parameter_block_idx中被保留的优化变量
    std::vector<double*> keep_block_data;  // 按顺序存放parameter_block_data中被保留的优化变量

    Eigen::MatrixXd linearized_jacobians;  // 边缘化得到的Jacobian矩阵
    Eigen::VectorXd linearized_residuals;  // 边缘化得到的残差
    const double eps = 1e-8;
};

class MarginalizationFactor : public ceres::CostFunction {
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

    MarginalizationInfo* marginalization_info;
};
