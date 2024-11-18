#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <filesystem>
#include "../../hnswlib/npy.hpp"

int main() {
    // 参数设置
    size_t start = 10000;
    size_t end = 100000;
    size_t step = 10000;
    size_t num_rows = 10000;
    std::string base_path = "/ssddata/ytianbc/test/marco_embeddings/";

    // 加载第一个文件以获取数据维度
    std::string first_file = base_path + "similarities_labels_" + std::to_string(start) + ".npy";
    std::vector<std::vector<float>> first_data;
    std::vector<unsigned long> shape_first;
    bool fortran_order_first;
    npy::LoadArrayFromNumpy(first_file, shape_first, fortran_order_first, first_data);

    // 生成随机行索引
    std::vector<size_t> indices(num_rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    indices.resize(num_rows);

    // 读取并拼接所有的 .npy 文件
    std::vector<std::vector<float>> data;
    for (size_t i = start; i <= end; i += step) {
        std::string file = base_path + "similarities_labels_" + std::to_string(i) + ".npy";
        std::vector<std::vector<float>> arr;
        std::vector<unsigned long> shape;
        bool fortran_order;
        npy::LoadArrayFromNumpy(file, shape, fortran_order, arr);

        // 选择随机的行
        std::vector<std::vector<float>> selected_rows(num_rows, std::vector<float>(shape[1]));
        for (size_t j = 0; j < num_rows; ++j) {
            selected_rows[j] = arr[indices[j]];
        }

        // 拼接数据
        if (data.empty()) {
            data = selected_rows;
        } else {
            for (size_t j = 0; j < num_rows; ++j) {
                data[j].insert(data[j].end(), selected_rows[j].begin(), selected_rows[j].end());
            }
        }
    }

    // 获取数据维度
    size_t n = data.size();
    size_t m = data[0].size();

    // 生成二进制掩码矩阵
    std::vector<std::vector<int>> mask(n, std::vector<int>(m, 0));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(0.2);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            mask[i][j] = dist(gen);
        }
    }

    // 设置第一列全为1
    for (size_t i = 0; i < n; ++i) {
        mask[i][0] = 1;
    }

    // 保存文件
    npy::SaveArrayAsNumpy("mask_gene.npy", mask, false);
    npy::SaveArrayAsNumpy("data_gene.npy", data, false);

    std::vector<size_t> indices_as_vector(indices.begin(), indices.end());
    npy::SaveArrayAsNumpy("indices_gene.npy", indices_as_vector, false);

    std::cout << "Data saved successfully!" << std::endl;

    return 0;
}
