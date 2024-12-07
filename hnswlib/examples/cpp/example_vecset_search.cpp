#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <omp.h>
#include<complex>
#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/space_l2.h"
#include "../../hnswlib/vectorset.h"
#include "../../cnpy/cnpy.h"

constexpr int VECTOR_DIM = 128;
constexpr int BASE_VECTOR_SET_MIN = 36;
constexpr int BASE_VECTOR_SET_MAX = 48;
constexpr int NUM_BASE_SETS = 10000;
constexpr int NUM_QUERY_SETS = 10;
constexpr int QUERY_VECTOR_COUNT = 32;
constexpr int K = 10;

class GroundTruth {
public:
    void build(int d, const std::vector<vectorset>& base) {
        dimension = d;
        base_vectors = base;
    }

    void search(const vectorset query, int k, std::vector<std::pair<int, float>>& res) const {
        res.clear();
        std::vector<std::pair<float, int>> distances;

        // Calculate Chamfer distance between query set and each base set
        //std::cout<<"Calc Dis"<<std::endl;
        int base_offset = 0;
        for (size_t i = 0; i < base_vectors.size(); ++i) {
            float chamfer_dist = hnswlib::L2SqrVecSet(&query, &base_vectors[i]);
            distances.push_back({chamfer_dist, static_cast<int>(i)});
        }
        //std::cout<<"return ans"<<std::endl;
        // Sort distances to find top-k nearest neighbors
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        for (int i = 0; i < k; ++i) {
            res.push_back(std::make_pair(distances[i].second, distances[i].first));
        }
        
    }

private:
    int dimension;
    std::vector<vectorset> base_vectors;
};

class Solution {
public:
    void build(int d, const std::vector<vectorset>& base) {
        dimension = d;
        base_vectors = base;
        space_ptr = new hnswlib::L2VSSpace(dimension);
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(space_ptr, NUM_BASE_SETS, 16, 32);
        #pragma omp parallel for schedule(dynamic)
        for(hnswlib::labeltype i = 0; i < base_vectors.size(); i++){
            if (i % 100 == 0) {
                std::cout << i << std::endl;
            }
            alg_hnsw->addPoint(&base_vectors[i], i);            
        }
        // Add any necessary pre-computation or indexing for the optimized search
    }

    void search(const vectorset query, int k, std::vector<std::pair<int, float>>& res) const {
        res.clear();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(&query, k);
        for(int i = 0; i < k; i++){
            res.push_back(std::make_pair(result.top().second, result.top().first));
            result.pop();
        }
    }

private:
    int dimension;
    std::vector<vectorset> base_vectors;
    hnswlib::L2VSSpace* space_ptr;
    hnswlib::HierarchicalNSW<float>* alg_hnsw;
};


float half_to_float(uint16_t h) {
    // 参考 IEEE 754 半精度转换公式
    int s = (h >> 15) & 0x1;                   // 符号位
    int e = (h >> 10) & 0x1F;                  // 指数部分
    int f = h & 0x3FF;                         // 尾数部分
    if (e == 0) {                              // 次正规数
        return (s ? -1 : 1) * std::ldexp(f, -24);
    } else if (e == 31) {                      // 特殊值（NaN 或 Infinity）
        return (s ? -1 : 1) * (f ? NAN : INFINITY);
    } else {                                   // 规范化数
        return (s ? -1 : 1) * std::ldexp(f + 1024, e - 15 - 10);
    }
}




void load_from_msmarco(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query, 
                       int file_numbers) {

    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = "/ssddata/0.6b_128d_dataset/encoding" + std::to_string(i) + "_float16.npy";
        std::string lensfile_name = "/ssddata/0.6b_128d_dataset/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];
        // int* lens_data = lens_npy.data<int>();
        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = lens_npy.shape[0];
        int offset = 0;
        std::cout << arr_npy.shape[0] << " " << arr_npy.shape[1] << " " << num_elements << std::endl;
        std::cout << doc_num << std::endl;
        std::cout << lens_npy.word_size << std::endl;
        assert (doc_num == 25000);
        
        for (size_t i = 0; i < num_elements; ++i) {
            base_data.push_back(static_cast<float>(half_to_float(raw_vec_data[i])));
        }
        
        for (int i = 0; i < doc_num; ++i) {
            base.push_back(vectorset(base_data.data() + offset, VECTOR_DIM, lens_data[i].real()));
            offset += lens_data[i].real() * VECTOR_DIM;
        }
    }

    std::string qembfile_name = "/ssddata/0.6b_128d_dataset/qembs_6980.npy";
    std::string qlensfile_name = "/ssddata/0.6b_128d_dataset/qlens_6980.npy";
    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);
    cnpy::NpyArray qlens_npy = cnpy::npy_load(qlensfile_name);

    uint16_t* raw_qembs_data = qembs_npy.data<uint16_t>();
    size_t num_qembs_elements = qembs_npy.shape[0] * qembs_npy.shape[1];

    std::complex<int>* qlens_data = qlens_npy.data<std::complex<int>>();
    size_t q_num = qlens_npy.shape[0];

    int q_offset = 0;
    assert (q_num == 6980);
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data.push_back(static_cast<float>(half_to_float(raw_qembs_data[i])));
    }
    
    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, VECTOR_DIM, qlens_data[i].real()));
        q_offset += qlens_data[i].real() * VECTOR_DIM;
    }
    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << std::endl;
}

void generate_vector_sets(std::vector<float>& base_data, std::vector<vectorset>& base,
                          std::vector<float>& query_data, std::vector<vectorset>& query, 
                          int num_base_sets, int num_query_sets) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    std::uniform_int_distribution<int> vec_count_dis(BASE_VECTOR_SET_MIN, BASE_VECTOR_SET_MAX);

    // Step 1: Generate base vector data
    std::vector<int> base_vec_counts;  // Track the number of vectors for each base set
    for (int i = 0; i < num_base_sets; ++i) {
        int num_vectors = vec_count_dis(gen);
        // int num_vectors = i + 1;
        base_vec_counts.push_back(num_vectors);
        
        // Fill base_data with random values for each vector
        for (int j = 0; j < num_vectors; ++j) {
            for (int d = 0; d < VECTOR_DIM; ++d) {
                base_data.push_back(dis(gen));
            }
        }
    }

    // Step 2: Create vectorset objects for the base vector sets
    int offset = 0;
    for (int i = 0; i < num_base_sets; ++i) {
        base.push_back(vectorset(base_data.data() + offset, VECTOR_DIM, base_vec_counts[i]));
        offset += base_vec_counts[i] * VECTOR_DIM;
    }

    // Step 3: Generate query vector data
    for (int i = 0; i < num_query_sets; ++i) {
        // Fill query_data with random values for each vector
        for (int j = 0; j < QUERY_VECTOR_COUNT; ++j) {
            for (int d = 0; d < VECTOR_DIM; ++d) {
                query_data.push_back(dis(gen));
            }
        }
    }

    // Step 4: Create vectorset objects for the query vector sets
    offset = 0;
    for (int i = 0; i < num_query_sets; ++i) {
        query.push_back(vectorset(query_data.data() + offset, VECTOR_DIM, QUERY_VECTOR_COUNT));
        offset += QUERY_VECTOR_COUNT * VECTOR_DIM;
    }

    // Debug output to confirm correct data range
    std::cout << "Sample value from base_data: " << base_data[16] << std::endl;
}

double calculate_recall(const std::vector<std::pair<int, float>>& solution_indices,
                        const std::vector<std::pair<int, float>>& ground_truth_indices) {
    std::unordered_set<int> solution_set;
    std::unordered_set<int> ground_truth_set;
    for (const auto& pair : solution_indices) {
        solution_set.insert(pair.first);
    }
    for (const auto& pair : ground_truth_indices) {
        ground_truth_set.insert(pair.first);
    }
    int intersection_count = 0;
    for (const int& index : solution_set) {
        if (ground_truth_set.find(index) != ground_truth_set.end()) {
            intersection_count++;
        }
    }

    double recall = static_cast<double>(intersection_count) / ground_truth_set.size();
    return recall;
}

int main() {
    std::vector<float> base_data;
    std::vector<int> base_vec_num;
    std::vector<float> query_data;
    std::vector<vectorset> base;
    std::vector<vectorset> query;
    // Generate dataset
    generate_vector_sets(base_data, base, query_data, query, NUM_BASE_SETS, NUM_QUERY_SETS);
    // load_from_msmarco(base_data, base, query_data, query, 1);

    // std::cout<<"outliner"<< std::endl;
    // for (int i = 0; i<base.size(); i++){
    //     for(int j = 0; j < base[i].dim * base[i].vecnum; j++) {
    //         if(*(base[i].data + j) > 1 ||  *(base[i].data + j) < -1)
    //             std::cout << "outliner" << *(base[i].data + j) << std::endl;
    //     }
    // }
    GroundTruth ground_truth;
    ground_truth.build(VECTOR_DIM, base);
    std::cout<< "Generate Groundtruth and Dataset" <<std::endl;
    Solution solution;
    solution.build(VECTOR_DIM, base);
    double total_recall = 0.0;

    std::cout<<"Processing Queries"<<std::endl;
    for (int i = 0; i < NUM_QUERY_SETS; ++i) {
        std::vector<std::pair<int, float>> ground_truth_indices, solution_indices;
        // Search with GroundTruth
        ground_truth.search(query[i], K, ground_truth_indices);
        std::cout<<"BruteForce Result: ";
        for(int j = 0 ; j < ground_truth_indices.size(); j++)
            std::cout<<ground_truth_indices[j].first<<":"<<ground_truth_indices[j].second<<std::endl;
        std::cout<<std::endl;
        // Search with Solution
        solution.search(query[i], K, solution_indices);
        std::cout<<"HNSW Result: ";
        for(int j = 0 ; j < solution_indices.size(); j++)
            std::cout<<solution_indices[j].first<<":"<<solution_indices[j].second<<std::endl;
        std::cout<<std::endl<<std::endl;
        // Calculate recall for this query set
        double recall = calculate_recall(solution_indices, ground_truth_indices);
        total_recall += recall;
        std::cout << "Recall for query set " << i << ": " << recall << std::endl;
    }

    // Calculate average recall
    double average_recall = total_recall / NUM_QUERY_SETS;
    std::cout << "Average Recall: " << average_recall << std::endl;

    return 0;
}
