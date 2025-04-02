#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <omp.h>
#include <complex>
#include <cblas.h>
#include <chrono>  
#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/space_l2.h"
#include "../../hnswlib/vectorset.h"
#include "../../cnpy/cnpy.h"

constexpr int VECTOR_DIM = 128;
constexpr int BASE_VECTOR_SET_MIN = 36;
constexpr int BASE_VECTOR_SET_MAX = 48;
constexpr int MSMACRO_TEST_NUMBER = 354;
constexpr int NUM_BASE_SETS = 25000 * MSMACRO_TEST_NUMBER;
constexpr long long NUM_BASE_VECTOR_LOTTE = 339419977;
constexpr int NUM_BASE_SETS_LOTTE = 2428853;
constexpr int NUM_QUERT_LOTTE = 2930;
constexpr int LOTTE_TEST_NUMBER = 98;
int NUM_QUERY_SETS = 100;
constexpr int QUERY_VECTOR_COUNT = 32;
constexpr int K = 100;
constexpr int NUM_CLUSTER = 262144;
constexpr int NUM_GRAPH_CLUSTER = 256;

void convert_to_column_major(const std::vector<float>& row_major_matrix, std::vector<float>& col_major_matrix, int rows, int cols) {
    col_major_matrix.resize(rows * cols); // 重新分配空间

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            col_major_matrix[j * rows + i] = row_major_matrix[i * cols + j];  // 交换行列
        }
    }
}

void topk_avx512(const std::vector<float>& matrix, int rows, int cols, int topk, std::unordered_set<int>& unique_indices) {
    // #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {  // 遍历 32 行
        __m512 top_val = _mm512_set1_ps(-1e30f);  // 存储 top-4 最大值
        __m512i top_idx = _mm512_set1_epi32(-1);  // 存储索引
        __m512i indices = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

        for (int j = 0; j + 15 < cols; j += 16) {  // 每次处理 16 列
            __m512 data = _mm512_loadu_ps(&matrix[i * cols + j]);  // 读取 16 个 float
            __m512i index = _mm512_add_epi32(_mm512_set1_epi32(j), indices);  // 计算索引

            // 选出 top-4 值
            __mmask16 mask = _mm512_cmp_ps_mask(data, top_val, _CMP_GT_OQ);
            top_val = _mm512_mask_blend_ps(mask, top_val, data);
            top_idx = _mm512_mask_blend_epi32(mask, top_idx, index);
        }

        // 提取 top-4 值和索引
        float top_values[16];
        int top_indices[16];
        _mm512_storeu_ps(top_values, top_val);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(top_indices), top_idx);

        // 仅取 top-4
        std::vector<std::pair<float, int>> top_pairs;
        for (int k = 0; k < 16; ++k) {
            if (top_indices[k] < cols) {  // 确保索引合法
                top_pairs.emplace_back(top_values[k], top_indices[k]);
            }
        }
        if (top_pairs.size() > topk) {
            std::partial_sort(top_pairs.begin(), top_pairs.begin() + topk, top_pairs.end(),
                            std::greater<std::pair<float, int>>());

            for (int k = 0; k < topk; ++k) {
                unique_indices.insert(top_pairs[k].second);
            }
        } else {
            for (int k = 0; k < top_pairs.size(); ++k) {
                unique_indices.insert(top_pairs[k].second);
            }            
        }
    }
    return;
}

void get_unique_top_k_indices_col(const std::vector<float>& matrix, int rows, int cols, int topk, std::unordered_set<int>& unique_indices) {
    // std::cout << matrix.size() << std::endl;
    std::vector<int> all_scores(rows * topk);
    // #pragma omp parallel for
    // std::cout << matrix.size() << std::endl;
    // std::cout << rows << std::endl;
    // std::cout << cols << std::endl;
    // std::cout << topk << std::endl;
    // std::cout << all_scores.size() << std::endl;
    #pragma omp parallel for
    for (int row = 0; row < rows; ++row) {
        // std::cout << row << std::endl;
        std::vector<std::pair<float, int>> scores(cols);
        // 提取该列数据
        for (int col = 0; col < cols; ++col) {
            // std::cout << row << " " << col << std::endl;
            scores[col] = {matrix[row * cols + col], col};  
        }

        // 仅排序前 K 个最大元素
        std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                              return a.first > b.first; // 降序
                          });

        // 直接存入集合去重
        for (int i = 0; i < topk; ++i) {
            // std::cout << row << " " << row * topk + i << " " << scores[i].first << " " << scores[i].second << std::endl;
            all_scores[row * topk + i] = scores[i].second;
        }
    }
    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < topk; ++i) {
            // std::cout << row << " " << i << std::endl;
            unique_indices.insert(all_scores[row * topk + i]);
        }
    }
    // for(int t: unique_indices) {
    //     std::cout<< t << " ";
    // }
    // std::cout << std::endl;
    return;
}

std::unordered_set<int> get_unique_top_k_indices(const std::vector<float>& matrix, int rows, int cols, int k) {
    std::unordered_set<int> unique_indices;  // 存储去重索引

    for (int col = 0; col < cols; ++col) {
        std::vector<std::pair<float, int>> scores(rows);

        // 提取该列数据
        for (int row = 0; row < rows; ++row) {
            scores[row] = {matrix[row * cols + col], row};  // 行优先存储
        }

        // 仅排序前 K 个最大元素
        std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                              return a.first > b.first; // 降序
                          });

        // 直接存入集合去重
        for (int i = 0; i < k; ++i) {
            unique_indices.insert(scores[i].second);
        }
    }

    return unique_indices;
}

class GroundTruth {
public:
    void build(int d, const std::vector<vectorset>& base) {
        dimension = d;
        base_vectors = base;
    }


    void search(const vectorset query, int k, std::vector<std::pair<int, float>>& res) const {
        // res.clear();
        res.clear();
        std::vector<std::pair<float, int>> distances;
        std::priority_queue<std::pair<float, int>> max_heap;
        // Calculate Chamfer distance between query set and each base set
        //std::cout<<"Calc Dis"<<std::endl;
        int base_offset = 0;
        for (size_t i = 0; i < base_vectors.size(); ++i) {
            // if (i % 1000000 == 0) {
            //     std::cout << "calc distance for " << i << std::endl;
            // }
            float chamfer_dist = hnswlib::L2SqrVecSet(&query, &base_vectors[i], 0);
            // distances.push_back({chamfer_dist, static_cast<int>(i)});
            // 如果堆的大小小于 k，直接插入
            if (max_heap.size() < static_cast<size_t>(k)) {
                max_heap.emplace(chamfer_dist, static_cast<int>(i));
            } 
            // 如果当前距离比堆顶小，替换堆顶
            else if (chamfer_dist < max_heap.top().first) {
                max_heap.pop();
                max_heap.emplace(chamfer_dist, static_cast<int>(i));
            }
        }
        //std::cout<<"return ans"<<std::endl;
        // Sort distances to find top-k nearest neighbors
        while (!max_heap.empty()) {
            res.emplace_back(max_heap.top().second, max_heap.top().first);
            max_heap.pop();
        }

        // 按距离从小到大排序
        std::sort(res.begin(), res.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second < b.second;
        });
    }

    void searchCluster(const vectorset query, const vectorset query_cluster, int k, std::vector<std::pair<int, float>>& res) const {
        // res.clear();
        res.clear();
        std::vector<std::pair<float, int>> distances;
        std::priority_queue<std::pair<float, int>> max_heap;
        int base_offset = 0;
        for (size_t i = 0; i < base_vectors.size(); ++i) {
            float chamfer_dist = hnswlib::L2SqrCluster4Search(&query_cluster, &base_vectors[i], 0);
            if (max_heap.size() < static_cast<size_t>(k)) {
                max_heap.emplace(chamfer_dist, static_cast<int>(i));
            } 
            else if (chamfer_dist < max_heap.top().first) {
                max_heap.pop();
                max_heap.emplace(chamfer_dist, static_cast<int>(i));
            }
        }
        while (!max_heap.empty()) {
            res.emplace_back(max_heap.top().second, max_heap.top().first);
            max_heap.pop();
        }
        // 按距离从小到大排序
        std::sort(res.begin(), res.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second < b.second;
        });
    }

private:
    int dimension;
    std::vector<vectorset> base_vectors;
};

class Solution {
public:
    void build(int d, const std::vector<vectorset>& base, const std::vector<std::vector<hnswlib::labeltype>>& cluster_set, const std::vector<int>& temp, const std::vector<float>& cluster_distance) {
        double time = omp_get_wtime();
        alg_hnsw_list.resize(NUM_GRAPH_CLUSTER);
        dimension = d;
        base_vectors = base;
        space_ptr = new hnswlib::L2VSSpace(dimension);
        temp_cluster_id = temp;
        std::cout << "init alg" << std::endl;
        // omp_set_num_threads(160);
        // #pragma omp parallel for schedule(dynamic) num_threads(6)
        for(int tmpi = 0; tmpi < temp_cluster_id.size(); tmpi++) {
            int i = temp_cluster_id[tmpi];
            double cur_time = omp_get_wtime();
            std::cout << "cluster build begin: " + std::to_string(i) + " " + std::to_string(cluster_set[i].size()) << std::endl;
            alg_hnsw_list[i] = new hnswlib::HierarchicalNSW<float>(space_ptr, cluster_set[i].size() + 1, 16, 80);
            // #pragma omp parallel for schedule(dynamic)
            // #pragma omp parallel for schedule(dynamic, 512)
            #pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < cluster_set[i].size(); j++) {
                if (j % 1000 == 0) {
                    // #pragma omp critical
                    std::cout << std::to_string(i) + " " + std::to_string(j) << std::endl;
                }
                alg_hnsw_list[i]->addClusterPoint(&base_vectors[cluster_set[i][j]], cluster_distance.data(), cluster_set[i][j]);
            }
            std::cout << "cluster build finish: " + std::to_string(i) + " " + std::to_string(omp_get_wtime() - cur_time) << std::endl;
        }
        std::cout << "Build time: " << omp_get_wtime() - time << "sec"<<std::endl;
        // Add any necessary pre-computation or indexing for the optimized search
    }

    // double search(const vectorset query, int k, int ef, std::vector<std::pair<int, float>>& res) const {
    //     res.clear();
    //     alg_hnsw->setEf(ef);
    //     double start_time = omp_get_wtime();
    //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnFineEdge(&query, k);
    //     // std::cout << alg_hnsw->metric_hops << ' ' << alg_hnsw->metric_distance_computations << std::endl;
    //     // alg_hnsw->metric_hops = 0;
    //     // alg_hnsw->metric_distance_computations = 0;
    //     for(int i = 0; i < k; i++){
    //         res.push_back(std::make_pair(result.top().second, result.top().first));
    //         result.pop();
    //     }
    //     double end_time = omp_get_wtime();
    //     return end_time - start_time;
    // }

    double search_with_cluster(const vectorset& query, std::vector<float>& query_cluster_scores, std::vector<float>& col_query_cluster_scores, std::vector<float>& center_data, std::vector<float>& graph_center_data, int k, int ef, std::vector<std::pair<int, float>>& res) {
        // std::cout << " stage-2 ";
        std::vector<float> graph_cluster_scores(NUM_GRAPH_CLUSTER * 32);
        res.clear();
        // alg_hnsw_list[0]->setEf(ef);
        // std::cout << " stage-1 ";
        // l2_vec_call_count.store(0);
        // std::cout << " stage-1 ";
        // std::vector<float> col_query_cluster_scores(262144 * 32);
        // std::cout << " stage0 ";
        double start_time = omp_get_wtime();
        // query_cluster_scores.resize(NUM_QUERY_SETS * 262144 * 32);
        // for (int j = 0; j < 32; j++) {
        //     for (int k = 0; k < 262144; k++) {
        //         float tt  = hnswlib::InnerProductDistance((&query)->data + j * 128, &center_data[k * 128], &(&query)->dim);
        //         query_cluster_scores[j * 262144 + k] =  tt;
        //     }
        // }
        // hnswlib::fast_dot_product_blas(262144, 128, 32, center_data.data(), (&query)->data, query_cluster_scores.data());
        // std::cout << " stage0 ";  
        hnswlib::fast_dot_product_blas(32, 128, NUM_GRAPH_CLUSTER, (&query)->data, graph_center_data.data(), graph_cluster_scores.data()); 
        // std::cout << " stage1 ";
        double start_time_2 = omp_get_wtime();
        std::unordered_set<int> unique_indices;
        // unique_indices.reserve(256);
        // std::cout << graph_cluster_scores[0] << " " << graph_cluster_scores[1] << " " << graph_cluster_scores[2] << std::endl;
        get_unique_top_k_indices_col(graph_cluster_scores, 32, NUM_GRAPH_CLUSTER, 50, unique_indices); 
        // unique_indices.insert(182);
        // std::vector<int> unique_indices = {79, 85, 126, 182, 225, 245};
        // std::cout << unique_indices.size() << std::endl;
        // unique_indices.insert(182);
        // std::cout << " stage2 ";
        double start_time_3 = omp_get_wtime();
        // convert_to_column_major(query_cluster_scores, col_query_cluster_scores, 32, 262144);
        hnswlib::fast_dot_product_blas(262144, 128, 32, center_data.data(), (&query)->data, col_query_cluster_scores.data());  
        // std::cout << col_query_cluster_scores[0] << std::endl;
        vectorset query_cluster = vectorset(col_query_cluster_scores.data(), nullptr, 262144, 32);
        // std::cout << " stage3 ";
        double start_time_4 = omp_get_wtime();
        // std::cout << start_time_4 << std::endl;
        double cluster_time = omp_get_wtime();
        // std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnFineEdge(&query, k);
        std::unordered_set<hnswlib::labeltype> search_result;
        // std::cout << " " << unique_indices.size() << " ";
        // std::cout << " stage4 ";
        std::vector<std::pair<float, hnswlib::labeltype>> merge_result;
        for (const int idx : unique_indices) {
            // std::cout << " " << idx << std::endl;
            if (alg_hnsw_list[idx] == nullptr) {
                continue;
            }
            alg_hnsw_list[idx]->setEf(ef);
            // std::cout << " " << idx << " " << std::endl;
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw_list[idx]->searchKnnCluster(&query_cluster, ef);
            // std::cout << result.size() << std::endl;
            while(result.size() > 0) {
                if (search_result.find(result.top().second) == search_result.end()) {
                    search_result.insert(result.top().second);
                    merge_result.push_back(result.top());
                }
                result.pop();
            }
            // break;
        }
        // std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw_list[0]->searchKnnCluster(&query_cluster, ef);
        // std::cout << alg_hnsw->metric_hops << ' ' << alg_hnsw->metric_distance_computations << std::endl;
        // alg_hnsw->metric_hops = 0;
        // alg_hnsw->metric_distance_computations = 0;
        // std::cout << search_result.size() << " " << merge_result.size() << std::endl;
        double search_time = omp_get_wtime();
        int numrerank = std::min(merge_result.size(), 256ul);
        std::partial_sort(merge_result.begin(), merge_result.begin() + numrerank, merge_result.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first < b.first;  // 按 float 排序，越小越靠前
                      });
        // merge_result.resize(256);
        // std::cout << merge_result.size() << std::endl;
        for (int i = 0; i < numrerank; i++){
            // for (const hnswlib::labeltype ind : search_result){
            hnswlib::labeltype ind = merge_result[i].second;
            res.push_back(std::make_pair(ind, hnswlib::L2SqrVecCF(&query, &base_vectors[ind], 0)));
        }
        // std::cout << res.size() << std::endl;
        std::partial_sort(res.begin(), res.begin() + k, res.end(),
                      [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                          return a.second < b.second;  // 按 float 排序，越小越靠前
                      });
        res.resize(k);
        // std::cout << res.size() << std::endl;
        double end_time = omp_get_wtime();
        // std::cout << start_time_2 - start_time << " " << start_time_3 - start_time_2 << " " << start_time_4 - start_time_3 << std::endl;
        std::cout << "cluster calc: " << start_time_2 - start_time << " cluster find: " << start_time_3 - start_time_2 << " transfer col storage: " << start_time_4 - start_time_3  << " search time: " << search_time - cluster_time << " search calc count: " << l2_vec_call_count.load() << " rerank time: " << end_time - search_time << std::endl;
        return end_time - start_time;
    }

    void save(const std::string &location) {
        double time = omp_get_wtime();
        for(int tmpi = 0; tmpi < temp_cluster_id.size(); tmpi++) {
            int i = temp_cluster_id[tmpi];
            std::string locai = location + std::to_string(i) + ".bin";
            alg_hnsw_list[i]->saveIndex(locai);
        }
        // alg_hnsw->saveIndex(location);
        std::cout << "save time: " << omp_get_wtime() - time << "sec"<<std::endl;
    }

    void load(const std::string &location, int d, const std::vector<vectorset>& base, const std::vector<std::vector<hnswlib::labeltype>>& cluster_set, const std::vector<int>& temp) {
        double time = omp_get_wtime();
        dimension = d;
        base_vectors = base;
        space_ptr = new hnswlib::L2VSSpace(d);
        temp_cluster_id = temp;
        alg_hnsw_list.resize(NUM_GRAPH_CLUSTER);
        // for(int tmpi = 0; tmpi < NUM_GRAPH_CLUSTER; tmpi++) {
        //     int i = tmpi;
        //     alg_hnsw_list[i] = new hnswlib::HierarchicalNSW<float>(space_ptr, 10, 16, 80);
        // }        
        for(int tmpi = 0; tmpi < temp_cluster_id.size(); tmpi++) {
            int i = temp_cluster_id[tmpi];
            alg_hnsw_list[i] = new hnswlib::HierarchicalNSW<float>(space_ptr, cluster_set[i].size() + 1, 16, 80);
            alg_hnsw_list[i]->loadIndex(location + std::to_string(i) + ".bin", space_ptr);
            for (int j = 0; j < cluster_set[i].size(); j++) {
                alg_hnsw_list[i]->loadDataAddress(&base_vectors[cluster_set[i][j]], cluster_set[i][j]);
            }
            std::cout << i << std::endl;
        }
        std::cout << "load time: " << omp_get_wtime() - time << "sec"<<std::endl;
    }

    // void addEdge(hnswlib::labeltype p1, hnswlib::labeltype p2) {
    //     alg_hnsw->mutuallyConnectTwoElement(p1, p2);
    //     alg_hnsw->mutuallyConnectTwoElement(p2, p1);
    // }

    // bool canAddEdge(hnswlib::labeltype p1) {
    //     return alg_hnsw->canAddEdge(p1);
    // }

private:
    int dimension;
    std::vector<vectorset> base_vectors;
    hnswlib::L2VSSpace* space_ptr;
    // hnswlib::HierarchicalNSW<float>* alg_hnsw;
    std::vector<int> temp_cluster_id;
    std::vector<hnswlib::HierarchicalNSW<float>*> alg_hnsw_list;
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

std::vector<float> generate_random_vector(int n, float min, float max) {
    std::random_device rd;
    std::default_random_engine gen(rd());  // 随机数生成器
    std::uniform_real_distribution<float> dis(min, max);  // 在[min, max]范围内生成随机数

    std::vector<float> random_vector(n);
    for (int i = 0; i < n; ++i) {
        random_vector[i] = dis(gen);  // 填充随机值
    }

    return random_vector;
}

std::vector<int> generate_random_sign_sequence(int n) {
    std::random_device rd;
    std::default_random_engine gen(rd());  // 随机数生成器
    std::uniform_int_distribution<int> dis(0, 1);  // 生成 0 或 1

    std::vector<int> sequence(n);
    for (int i = 0; i < n; ++i) {
        // 根据随机生成的 0 或 1，赋值 -1 或 +1
        sequence[i] = (dis(gen) == 0) ? -1 : 1;
    }

    return sequence;
}


void demo_test_msmarco(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query, 
                       int file_numbers, std::vector<std::vector<int>>& qrels) {
    int offset = 0;   
    int p_offset = 0;                 
    for (int i = 0; i < 1; i++) {
        std::string embfile_name = "/ssddata/0.6b_128d_dataset/encoding" + std::to_string(i) + "_float16.npy";
        std::string lensfile_name = "/ssddata/0.6b_128d_dataset/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];
        // int* lens_data = lens_npy.data<int>();
        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = 25000;
        std::cout << "Processing file " << i << std::endl;
        // assert (doc_num == 25000);
        
        for (size_t i = 0; i < num_elements; ++i) {
            base_data[i] = (static_cast<float>(half_to_float(raw_vec_data[i])));
        }
        
        for (int i = 0; i < doc_num; ++i) {
            // std::cout << base_data[offset] << " " << lens_data[i].real() << std::endl;
            base.push_back(vectorset(base_data.data() + offset, VECTOR_DIM, lens_data[i].real()));
            offset += lens_data[i].real() * VECTOR_DIM;
        }    
    }

    std::string qembfile_name = "/ssddata/0.6b_128d_dataset/queries_embeddings.npy";
    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);
    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = NUM_QUERY_SETS * 32 * 128;

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data.push_back(static_cast<float>((raw_qembs_data[i])));
    }
    
    for (int i = 0; i < NUM_QUERY_SETS; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, VECTOR_DIM, 32));
        q_offset += 32 * VECTOR_DIM;
    }

    // std::string qembfile_name = "/ssddata/0.6b_128d_dataset/qembs_6980.npy";
    // std::string qlensfile_name = "/ssddata/0.6b_128d_dataset/qlens_6980.npy";
    // cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);
    // cnpy::NpyArray qlens_npy = cnpy::npy_load(qlensfile_name);

    // uint16_t* raw_qembs_data = qembs_npy.data<uint16_t>();
    // size_t num_qembs_elements = qembs_npy.shape[0] * qembs_npy.shape[1];

    // std::complex<int>* qlens_data = qlens_npy.data<std::complex<int>>();
    // size_t q_num = qlens_npy.shape[0];

    // int q_offset = 0;
    // assert (q_num == 6980);
    
    // for (size_t i = 0; i < num_qembs_elements; ++i) {
    //     query_data.push_back(static_cast<float>(half_to_float(raw_qembs_data[i])));
    // }
    
    // for (int i = 0; i < NUM_QUERY_SETS; ++i) {
    //     query.push_back(vectorset(query_data.data() + q_offset, VECTOR_DIM, qlens_data[i].real()));
    //     q_offset += qlens_data[i].real() * VECTOR_DIM;
    // }

    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
}



void load_from_msmarco(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query,
                       std::vector<int>& base_data_codes, std::vector<float>& center_data,
                       std::vector<float>& graph_center_data,
                       std::vector<std::vector<hnswlib::labeltype>>& cluster_set, 
                       int file_numbers, std::vector<std::vector<int>>& qrels) {
    long long offset = 0;  
    long long all_elements = 0; 
    long long code_offset = 0;
    long long all_codes = 0;  
    std::string cembfile_name = "/home/zhoujin/vecDB_publi_data/0.6b_128d_dataset/new_center_code/centroids.npy";
    std::string gcembfile_name = "/home/zhoujin/project/forremove/VecSetSearch/256_cluster_centroids.npy";
    std::string qembfile_name = "/home/zhoujin/vecDB_publi_data/0.6b_128d_dataset/qembs_32_6980.npy";
    std::string qrelfile_name = "/home/zhoujin/vecDB_publi_data/0.6b_128d_dataset/qrels_6980.tsv";
    std::string cdocsfile_name =  "/home/zhoujin/project/forremove/VecSetSearch/cluster_data/256_cluster_info_tfidf_2.txt";  

    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = "/home/zhoujin/vecDB_publi_data/0.6b_128d_dataset/encoding" + std::to_string(i) + "_float16.npy";
        std::string codesfile_name = "/home/zhoujin/vecDB_publi_data/0.6b_128d_dataset/new_center_code/doc_codes_" + std::to_string(i) + ".npy";
        std::string lensfile_name = "/home/zhoujin/vecDB_publi_data/0.6b_128d_dataset/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray codes_npy = cnpy::npy_load(codesfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];

        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = lens_npy.shape[0];

        int32_t* codes_data = codes_npy.data<int32_t>();
        size_t num_codes = codes_npy.shape[0];
        // std::cout << codes_npy.word_size << std::endl;
        // std::cout << sizeof(int32_t) << std::endl;
        // std::cout << sizeof(int16_t) << std::endl;
        // std::cout << sizeof(int) << std::endl;
        std::cout << num_codes << std::endl;
        std::cout << all_codes << std::endl;
        std::cout << "Processing file " << i << std::endl;
        
        for (long long i = 0; i < num_elements; ++i) {
            base_data[all_elements + i] = (static_cast<float>(half_to_float(raw_vec_data[i])));
        }
        // std::cout << "?" << std::endl;
        for (long long i = 0; i < num_codes; ++i) {
            base_data_codes[all_codes + i] = static_cast<int>(codes_data[i]);
            if (i < 10) {
                // std::cout << i << " " << all_codes + i << std::endl;
                std::cout << codes_data[i] << " ";
                std::cout << base_data_codes[all_codes + i] << " ";
            }
        }
        std::cout << std::endl;

        all_elements += num_elements;
        all_codes += num_codes;
        
        for (int i = 0; i < doc_num; ++i) {
            base.push_back(vectorset(base_data.data() + offset, base_data_codes.data() + code_offset, VECTOR_DIM, lens_data[i].real()));
            offset += lens_data[i].real() * VECTOR_DIM;
            code_offset += lens_data[i].real();
        }
    }

    cnpy::NpyArray cembs_npy = cnpy::npy_load(cembfile_name);
    uint16_t* raw_cembs_data = cembs_npy.data<uint16_t>();
    size_t num_cembs_elements = cembs_npy.shape[0] * cembs_npy.shape[1];
    for (size_t i = 0; i < num_cembs_elements; ++i) {
        center_data[i] = (static_cast<float>(half_to_float(raw_cembs_data[i])));
    }

    cnpy::NpyArray gcembs_npy = cnpy::npy_load(gcembfile_name);
    float* raw_gcembs_data = gcembs_npy.data<float>();
    size_t num_gcembs_elements = gcembs_npy.shape[0] * gcembs_npy.shape[1];
    for (size_t i = 0; i < num_gcembs_elements; ++i) {
        graph_center_data[i] = (static_cast<float>((raw_gcembs_data[i])));
    }
    // std::cout << num_cembs_elements << std::endl;

    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);

    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = NUM_QUERY_SETS * qembs_npy.shape[1] * qembs_npy.shape[2];
    size_t q_num = NUM_QUERY_SETS;

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }
    
    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, nullptr, VECTOR_DIM, 32));
        q_offset += 32 * VECTOR_DIM;
    }
    qrels.resize(q_num + 1);

    std::ifstream file(qrelfile_name);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        if (iss >> num1 >> num2) {
            if (num1 < 0 || num1 >= q_num) {
                continue;
                // std::cerr << "?" << line << std::endl;
            } else {
                // std::cout << num1 << " " << num2 << std::endl;
                qrels[num1].push_back(num2);
            }
        }
    }
    file.close();

    std::ifstream codcsfile(cdocsfile_name);
    std::string cdocs_line;
    int lineid = 0;
    while (std::getline(codcsfile, cdocs_line)) { // 逐行读取
        std::istringstream iss(cdocs_line);  // 创建字符串流
        hnswlib::labeltype num1;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        while (iss >> num1) {
            cluster_set[lineid].push_back(num1);
        }
        if (lineid % 100 == 0) {
            std::cout << lineid << " " << cluster_set[lineid].size() << std::endl;
        }
        lineid++;
    }
    file.close();

    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
}



void load_from_lotte(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query, 
                       int file_numbers, std::vector<std::vector<int>>& qrels) {
    long long offset = 0;  
    long long all_elements = 0;   
    std::string qembfile_name = "/home/zhoujin/data/lotte_embeddings/pooled/lotte_pooled_dev_query.npy";
    std::string qrelfile_name = "/home/zhoujin/data/lotte/pooled/dev/qas.search.tsv";    

    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = "/home/zhoujin/data/lotte_embeddings/pooled/encoding" + std::to_string(i) + "_float16.npy";
        std::string lensfile_name = "/home/zhoujin/data/lotte_embeddings/pooled/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];
        // int* lens_data = lens_npy.data<int>();
        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = lens_npy.shape[0];
        std::cout << "Processing file " << i << std::endl;
        // assert (doc_num == 25000);
        // std::cout << num_elements << std::endl;
        
        for (long long i = 0; i < num_elements; ++i) {
            base_data[all_elements + i] = (static_cast<float>(half_to_float(raw_vec_data[i])));
        }
        all_elements += num_elements;
        
        for (int i = 0; i < doc_num; ++i) {
            base.push_back(vectorset(base_data.data() + offset, VECTOR_DIM, lens_data[i].real()));
            offset += lens_data[i].real() * VECTOR_DIM;
        }
    }

    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);

    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = NUM_QUERT_LOTTE * qembs_npy.shape[1] * qembs_npy.shape[2];
    size_t q_num = NUM_QUERT_LOTTE;

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }
    
    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, VECTOR_DIM, 32));
        q_offset += 32 * VECTOR_DIM;
    }
    qrels.resize(q_num + 1);

    std::ifstream file(qrelfile_name);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        if (iss >> num1 >> num2) {
            if (num1 < 0 || num1 >= q_num) {
                continue;
                // std::cerr << "?" << line << std::endl;
            } else {
                // std::cout << num1 << " " << num2 << std::endl;
                qrels[num1].push_back(num2);
            }
        }
    }
    file.close();

    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
}


void load_noise_query(std::vector<float>& query_data, std::vector<vectorset>& query, 
                      std::string& qembfile_name) {
    long long offset = 0;  
    long long all_elements = 0;   
    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);

    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = 100 * qembs_npy.shape[1] * qembs_npy.shape[2];
    size_t q_num = 100;

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }
    
    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, VECTOR_DIM, 32));
        q_offset += 32 * VECTOR_DIM;
    }

    std::cout << "load data finish! passage count: " << " query count: " << query.size() << std::endl;
}


void load_train_query(std::vector<float>& query_data, std::vector<vectorset>& query, 
                      std::vector<std::vector<int>>& qrels) {
    long long offset = 0;  
    long long all_elements = 0;   
    std::string qembfile_name = "/home/zhoujin/data/train_query.npy";
    std::string qrelfile_name = "/home/zhoujin/data/qrels.train.select.reorder.tsv";    

    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);

    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = qembs_npy.shape[0] * qembs_npy.shape[1] * qembs_npy.shape[2];
    size_t q_num = qembs_npy.shape[0];

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }
    
    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, VECTOR_DIM, 32));
        q_offset += 32 * VECTOR_DIM;
    }
    qrels.resize(q_num + 1);

    std::ifstream file(qrelfile_name);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        if (iss >> num1 >> num2) {
            if (num1 < 0 || num1 >= q_num) {
                continue;
                // std::cerr << "?" << line << std::endl;
            } else {
                // std::cout << num1 << " " << num2 << std::endl;
                qrels[num1].push_back(num2);
            }
        }
    }
    file.close();
    std::cout << "load train query finish! query count: " << query.size() << " " << qrels.size() << std::endl;
}

void subset_test_msmarco(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query, 
                       std::vector<std::vector<int>>& qrels) {
    int offset = 0;  
    int q_offset = 0; 
    // CPU1         
    // std::string docembs_filename = "/ssddata/0.6b_128d_dataset/msmacro_subset_100q_95kp/doc_embs.npy";  
    // std::string doclens_filename = "/ssddata/0.6b_128d_dataset/msmacro_subset_100q_95kp/doc_lens.npy";  
    // std::string qembs_filename = "/ssddata/0.6b_128d_dataset/msmacro_subset_100q_95kp/qembs.npy";      
    // std::string qrels_filename = "/ssddata/0.6b_128d_dataset/msmacro_subset_100q_95kp/qrels.tsv";

    // CPU5
    // std::string docembs_filename = "/ssddata/vecDB_publi_data/0.6b_128d_dataset/msmacro_subset_100q_95kp/doc_embs.npy";  
    // std::string doclens_filename = "/ssddata/vecDB_publi_data/0.6b_128d_dataset/msmacro_subset_100q_95kp/doc_lens.npy";  
    // std::string qembs_filename = "/ssddata/vecDB_publi_data/0.6b_128d_dataset/msmacro_subset_100q_95kp/qembs.npy";      
    // std::string qrels_filename = "/ssddata/vecDB_publi_data/0.6b_128d_dataset/msmacro_subset_100q_95kp/qrels.tsv";


    // CPU5
    std::string docembs_filename = "/home/zhoujin/vecDB_publi_data/msmacro_subset_100q_95kp/doc_embs.npy";  
    std::string doclens_filename = "/home/zhoujin/vecDB_publi_data/msmacro_subset_100q_95kp/doc_lens.npy";  
    std::string qembs_filename = "/home/zhoujin/vecDB_publi_data/msmacro_subset_100q_95kp/qembs.npy";      
    std::string qrels_filename = "/home/zhoujin/vecDB_publi_data/msmacro_subset_100q_95kp/qrels.tsv";

    cnpy::NpyArray docembs_npy = cnpy::npy_load(docembs_filename);
    cnpy::NpyArray doclens_npy = cnpy::npy_load(doclens_filename);
    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembs_filename);

    uint16_t* raw_docembs_data = docembs_npy.data<uint16_t>();
    size_t docembs_elements = docembs_npy.shape[0] * docembs_npy.shape[1];
    
    std::complex<int>* doclens_data = doclens_npy.data<std::complex<int>>();
    size_t doclens_elements = doclens_npy.shape[0];
    
    float* raw_qembs_data = qembs_npy.data<float>();
    size_t qembs_elements = qembs_npy.shape[0] * qembs_npy.shape[1] * qembs_npy.shape[2];

    int doc_num = doclens_elements;
    int q_num = qembs_npy.shape[0];
    int q_vec_len = qembs_npy.shape[1];
    // std::cout << "doc_num " << doclens_npy.shape[0] << std::endl;
    // std::cout << "q_num " << doclens_npy.shape[1] << std::endl;
    // std::cout << "q_vec_len " << q_vec_len << std::endl;

    for (size_t i = 0; i < docembs_elements; ++i) {
        base_data[i] = (static_cast<float>(half_to_float(raw_docembs_data[i])));
    }
    std::cout << "load doc float data: " << base_data.size() << std::endl;
    for (int i = 0; i < doc_num; ++i) {
        base.push_back(vectorset(base_data.data() + offset, VECTOR_DIM, doclens_data[i].real()));
        offset += doclens_data[i].real() * VECTOR_DIM;
    }   
    std::cout << "load doc data: " << base.size() << std::endl;
    for (size_t i = 0; i < qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }
    std::cout << "load query float data: " << query_data.size() << std::endl;
    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, VECTOR_DIM, q_vec_len));
        q_offset += q_vec_len * VECTOR_DIM;
    }
    std::cout << "load query data: " << query.size() << std::endl;
    qrels.resize(q_num + 1);
    std::ifstream file(qrels_filename);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        // std::cout << line << std::endl;
        // std::cout << "?" << std::endl;
        if (iss >> num1 >> num2) {
            if (num1 < 0 || num1 >= q_num) {
                continue;
                // std::cerr << "?" << line << std::endl;
            } else {
                // std::cout << num1 << " " << num2 << std::endl;
                qrels[num1].push_back(num2);
            }
        }
    }
    file.close();
    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
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

void readGroundTruth(const std::string& ground_truth_file, std::vector<std::vector<std::pair<int, float>>>& ground_truth_indices) {
    std::ifstream inFile(ground_truth_file);
    if (!inFile.is_open()) {
        std::cerr << "Error opening file: " << ground_truth_file << std::endl;
        return;
    }

    std::string line;
    int i = 0;
    while (std::getline(inFile, line)) {
        std::vector<std::pair<int, float>> query_results;
        std::istringstream line_stream(line);
        int index;
        float distance;
        int j = 0;
        while (line_stream >> index >> distance) {
            ground_truth_indices[i][j] = std::make_pair(index, distance); 
            j+=1;
        }
        i+=1;
    }

    inFile.close();
}

std::vector<int> generateEntriesIndex(int multi_entries_num, int n) {
    std::vector<int> numbers;
    numbers.resize(n - 50);
    for (int i = 50; i < n; ++i) {
        numbers[i - 50] = i;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(numbers.begin(), numbers.end(), gen);
    numbers.resize(multi_entries_num);
    return numbers;
}

double calculate_recall_for_msmacro(const std::vector<std::pair<int, float>>& solution_indices,
                        const std::vector<int>& ground_truth_indices) {
    std::unordered_set<int> solution_set;
    std::unordered_set<int> ground_truth_set;
    for (const auto& pair : solution_indices) {
        solution_set.insert(pair.first);
        // if (solution_set.size() >= K) {
        //     break;
        // }
    }
    for (const auto& pid : ground_truth_indices) {
        ground_truth_set.insert(pid);
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

double calculate_recall(const std::vector<std::pair<int, float>>& solution_indices,
                        const std::vector<std::pair<int, float>>& ground_truth_indices) {
    std::unordered_set<int> solution_set;
    std::unordered_set<int> ground_truth_set;
    // for (int j = 0; j < K; j ++) {
    //     std::cout << ground_truth_indices[j].first << " " << ground_truth_indices[j].second << " ";
    // }
    // std::cout << ground_truth_indices.size() << std::endl;
    
    for (const auto& pair : solution_indices) {
        solution_set.insert(pair.first);
        // std::cout << pair.first << " ";
        // if (solution_set.size() >= K) {
        //     break;
        // }
    }
    // std::cout << std::endl;

    for (const auto& pair : ground_truth_indices) {
        ground_truth_set.insert(pair.first);
        // std::cout << pair.first << " ";
        if (ground_truth_set.size() >= 1) {
            break;
        }
    }

    // for (int j = 0; j < K; j ++) {
    //     ground_truth_set.insert(ground_truth_indices[j].first);
    //     // std::cout << ground_truth_indices[j].first << " ";
    //     if (ground_truth_set.size() >= K) {
    //         break;
    //     }
    // }
    // std::cout << std::endl;
    int intersection_count = 0;
    for (const int& index : solution_set) {
        if (ground_truth_set.find(index) != ground_truth_set.end()) {
            intersection_count++;
        }
    }
    // std::cout << intersection_count << std::endl;
    // std::cout << ground_truth_set.size() << std::endl;
    double recall = static_cast<double>(intersection_count) / ground_truth_set.size();
    return recall;
}

double calculate_entry_recall(const std::vector<hnswlib::labeltype>& solution_indices,
                        const std::vector<std::pair<int, float>>& ground_truth_indices,
                        int topk = 100) {
    std::unordered_set<int> solution_set;
    std::unordered_set<int> ground_truth_set;
    for (const auto& ind : solution_indices) {
        solution_set.insert(ind);
    }
    for (const auto& pair : ground_truth_indices) {
        ground_truth_set.insert(pair.first);
        if (ground_truth_set.size() >= topk) {
            break;
        }
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
    omp_set_nested(1);

    std::vector<float> base_data;
    std::vector<int> base_vec_num;
    std::vector<float> query_data;
    std::vector<vectorset> base;
    std::vector<vectorset> query;
    std::vector<vectorset> query_center;
    std::vector<vectorset> query_center_col;
    std::vector<int> base_data_codes;
    std::vector<float> center_data;
    std::vector<float> graph_center_data;
    std::vector<float> query_cluster_scores;
    std::vector<float> query_cluster_scores_col;
    std::vector<float> test_query_cluster_scores;
    std::vector<std::vector<hnswlib::labeltype>> cluster_set;
    std::vector<std::vector<int>> qrels;
    std::vector<std::vector<std::pair<int, float>>> bf_ground_truth(
        6980, std::vector<std::pair<int, float>>(4000, {0, 0.0f})
    );
    std::vector<std::vector<std::pair<int, float>>> bf_ground_truth_cf(
        6980, std::vector<std::pair<int, float>>(4000, {0, 0.0f})
    );
    int dataset = 0;
    bool test_subset = false;
    bool load_bf_from_cache = true;
    bool rebuild = false;
    bool reconnect = false;
    int dist_metric = 1;
    int multi_entries_num = 40;
    int multi_entries_range = 100;
    std::mt19937 gen(42);                    // 使用Mersenne Twister引擎
    std::uniform_int_distribution<int> dist(1, std::numeric_limits<int>::max());
    std::string ground_truth_file, index_file;

    // std::vector<int> temp_cluster_id(185);
    // std::ifstream codcsfile("/home/zhoujin/project/forremove/VecSetSearch/256_cluster_id_nprob_1_query_100.txt");
    // std::vector<int> temp_cluster_id(2761);
    // std::ifstream codcsfile("/home/zhoujin/project/forremove/VecSetSearch/temp_cluster_id.txt");
    // std::string cdocs_line;
    // int lineid = 0;
    // while (std::getline(codcsfile, cdocs_line)) { // 逐行读取
    //     std::istringstream iss(cdocs_line);  // 创建字符串流
    //     int num1;
    //     char delimiter;                // 用于捕获 \t 分隔符
    //     // 读取两个整数，用 \t 作为分隔符
    //     while (iss >> num1) {
    //         temp_cluster_id[lineid] = num1;
    //         lineid += 1;
    //         std::cout << num1 << std::endl;
    //     }
    // }
    // codcsfile.close();
    std::vector<int> temp_cluster_id = {1, 2, 4, 5, 6, 10, 12, 13, 14, 15, 19, 20, 28, 34, 38, 39, 40, 41, 45, 52, 53, 54, 55, 61, 63, 66, 67, 68, 72, 77, 80, 84, 88, 89, 91, 99, 100, 110, 114, 117, 118, 121, 126, 132, 134, 135, 140, 142, 146, 153, 158, 159, 161, 164, 165, 167, 171, 172, 176, 181, 184, 185, 192, 200, 208, 210, 213, 214, 217, 218, 219, 225, 227, 230, 231, 232, 233, 235, 239, 242, 244, 247, 249, 252, 255};
    // std::vector<int> temp_cluster_id = {182};
    std::cout << temp_cluster_id.size() << std::endl;
    // std::vector<int> query182 = {0, 11, 22, 23, 40, 42, 53, 60, 70, 78, 82, 83};

    if (dataset == 0) {
        if (dist_metric == 0) {
            if (test_subset) {
                ground_truth_file = "../examples/caches/95k_ground_truth_bi_summax_l2_top100.txt";
                index_file = "../examples/localIndex/95k_bi_summax_l2.bin";
            } else {
                ground_truth_file = "../examples/caches/ground_truth_bi_summax_l2_top100.txt";
                index_file = "../examples/localIndex/8m_bi_summax_l2.bin";
            }
        } else if (dist_metric == 1) {
            if (test_subset) {
                ground_truth_file = "../examples/caches/95k_ground_truth_single_summax_l2_top100.txt";
                index_file = "../examples/localIndex/95k_single_summax_l2.bin";
            } else {
                ground_truth_file = "../examples/caches/ground_truth_cluster_ip_top4k.txt";
                index_file = "../examples/localIndex/8m_emd_l2.bin";
            }
        } else {
            if (test_subset) {
                ground_truth_file = "../examples/caches/95k_ground_truth_new_summax_l2_top100.txt";
                index_file = "../examples/localIndex/95k_new_summax_l2.bin";
            } else {
                ground_truth_file = "../examples/caches/ground_truth_new_summax_l2_top100.txt";
                index_file = "../examples/localIndex/8m_new_summax_l2.bin";
            }
        }
    } 
    else if (dataset == 1) {
        ground_truth_file = "../examples/caches/ground_truth_new_summax_l2_top100.txt";
        index_file = "../examples/localIndex/lotte_emd_l2.bin";
    }

    // Solution testsolution;
    // testsolution.load(index_file, VECTOR_DIM, base);

    if (dataset == 0) {
        if (test_subset) {
            // test on collected 95k msmacro subset
            base_data.resize((long long) 96000 * VECTOR_DIM * 80);
            query_data.resize((long long) NUM_QUERY_SETS * VECTOR_DIM * 32);
            subset_test_msmarco(base_data, base, query_data, query, qrels);     
        } else {
            // test on all msmacro dataset
            base_data.resize((long long) 25000 * MSMACRO_TEST_NUMBER * 128 * 80);
            query_data.resize((long long) NUM_QUERY_SETS * 128 * 32 + 1);
            base_data_codes.resize((long long) 25000 * MSMACRO_TEST_NUMBER * 80);
            center_data.resize((long long) 262144 * 128);
            graph_center_data.resize(NUM_GRAPH_CLUSTER * 128);
            cluster_set.resize(NUM_GRAPH_CLUSTER);
            // std::cout<< (long long) 25000 * MSMACRO_TEST_NUMBER * 80 << std::endl;
            load_from_msmarco(base_data, base, query_data, query, base_data_codes, center_data, graph_center_data, cluster_set, MSMACRO_TEST_NUMBER, qrels);
        }
    }
    else if (dataset == 1) {
        base_data.resize((long long) NUM_BASE_VECTOR_LOTTE * 128);
        query_data.resize((long long) NUM_QUERT_LOTTE * 128 * 32);
        load_from_lotte(base_data, base, query_data, query, LOTTE_TEST_NUMBER, qrels);
    }
    // std::vector<float> C(200 * 200);
    // double t0 = omp_get_wtime();
    // for (int i = 0; i < 100; i++) {
    //     float d = hnswlib::L2SqrVecBlasCF(&query[i], &base[i], C.data(), 0);
    //     // if (i < 10) std::cout << d << std::endl;
    // }
    // query_cluster_scores.resize(NUM_QUERY_SETS * 262144 * 32);
    // double t1 = omp_get_wtime();
    // hnswlib::L2SqrVecGetDistance((&query[0])->data, center_data.data(), query_cluster_scores.data(), 32, 262144, 128);    
    // double t2 = omp_get_wtime();
    // hnswlib::fast_dot_product_blas(32, 262144, 128, (&query[0])->data, center_data.data(), query_cluster_scores.data());    
    // double t3 = omp_get_wtime();
    // std::cout << t2 - t1 << " " << t3 - t2 << std::endl;
    // return 0;
    std::vector<float> col_query_cluster_scores(262144 * 32);
    test_query_cluster_scores.resize(262144 * 32);
    col_query_cluster_scores.resize(262144 * 32);
    // hnswlib::fast_dot_product_blas(32, 128, 262144, (&query[0])->data, center_data.data(), test_query_cluster_scores.data());  
    // query_cluster_scores.resize(NUM_QUERY_SETS * 262144 * 32);
    // query_cluster_scores_col.resize(NUM_QUERY_SETS * 262144 * 32);
    // for (int i = 0; i < NUM_QUERY_SETS; ++i) {
    //     for (int j = 0; j < 32; j++) {
    //         for (int k = 0; k < 262144; k++) {
    //             float tt  = hnswlib::InnerProductSIMD16ExtSSE((&query[i])->data + j * 128, &center_data[k * 128], &(&query[i])->dim);
    //             // if (i == 0 && (k < 100 || k > 262100)) {
    //             //     std::cout << i << " " << j <<  " " << k << " " << tt << std::endl;
    //             //     std::cout << test_query_cluster_scores[j * 262144 + k] << std::endl;
    //             // }
    //             query_cluster_scores[i * 262144 * 32 + j * 262144 + k] =  tt;
    //             query_cluster_scores_col[i * 262144 * 32 + k * 32 + j] = tt;
    //         }
    //     }
    //     query_center.push_back(vectorset(query_cluster_scores.data() + i * 262144 * 32, nullptr, 262144, 32));
    //     query_center_col.push_back(vectorset(query_cluster_scores_col.data() + i * 262144 * 32, nullptr, 262144, 32));
    // }

    // std::random_device rd; 
    // std::mt19937 gen_int(rd());  // 使用 Mersenne Twister 伪随机数生成器
    // std::uniform_int_distribution<int> dist_int(0, 25000);  // 设定范围 0 ~ 250000

    // // 生成 20000 个随机数
    // std::vector<int> random_numbers(20000);


    // std::cout << "begin test" << std::endl;

    // double all_time_original_cf = 0.0f;
    // double all_time_blas_cf = 0.0f;
    // double all_time_eigen_cf = 0.0f;
    // std::vector<std::vector<float>> res1(100);
    // std::vector<std::vector<float>> res2(100);
    // std::vector<std::vector<float>> res3(100);
    // std::vector<float> C(32 * 180);
    // for (int i = 0; i < NUM_QUERY_SETS; ++i) {
    //     for (int j = 0; j < 20000; j ++) {
    //         int num = dist_int(gen_int);
    //         random_numbers[j] = num;
    //     } 
    //     double t1_original_cf = omp_get_wtime();
    //     res1[i].resize(20000);
    //     for (int j = 0; j < 20000; j++) {
    //         float d1 = hnswlib::L2SqrVecCF(&query[i], &base[j], 0);
    //         res1[i][j] = d1;
    //     }
    //     // continue;
    //     double t2_original_cf = omp_get_wtime();
    //     double t1_blas_cf = omp_get_wtime();
    //     res2[i].resize(20000);
    //     for (int j = 0; j < 20000; j++) {
    //         float d = hnswlib::L2SqrVecBlasCF(&query[i], &base[j], C.data(), 0);
    //         res2[i][j] = (d);
    //     }
    //     double t2_blas_cf = omp_get_wtime();
    //     std::vector<float> tmpq(32*128);
    //     std::vector<float> tmpd(180*128);
    //     for (int j = 0; j < 32; j++) {
    //         for (int k = 0; k < 128; k++) {
    //             tmpq[k * 32 + j] = *((&query[i])->data + j * 128 + k);
    //         }
    //     }
    //     double t1_eigen_cf = omp_get_wtime();
    //     res3[i].resize(20000);
    //     for (int j = 0; j < 20000; j++) {
    //         // for (int t = 0; t < (&base[j])->vecnum; t++) {
    //         //     for (int k = 0; k < 128; k++) {
    //         //         tmpd[k * 32 + t] = *((&base[j])->data + t * 128 + k);
    //         //     }
    //         // }
    //         // float d = hnswlib::L2SqrVecEigenCF(&query[i], &base[j], 0);
    //         float d = hnswlib::max_inner_product_sum(tmpq.data(), (&base[j])->data, 32, (&base[j])->vecnum, 128) / 32;
    //         res3[i][j] = (d);
    //     }
    //     double t2_eigen_cf = omp_get_wtime();
    //     std::cout << i << ": original 2w:  " << t2_original_cf - t1_original_cf << " blas 2w: " << t2_blas_cf - t1_blas_cf << " eigen 2w: " << t2_eigen_cf - t1_eigen_cf << std::endl;
    //     all_time_original_cf += t2_original_cf - t1_original_cf;
    //     all_time_blas_cf += t2_blas_cf - t1_blas_cf;
    //     all_time_eigen_cf += t2_eigen_cf - t1_eigen_cf;
    // }
    // std::cout << res1.size() << std::endl;
    // std::cout << res2.size() << std::endl;
    // std::cout << res3.size() << std::endl;
    // // double t2 = omp_get_wtime();
    // std::cout << all_time_original_cf / NUM_QUERY_SETS << std::endl;
    // std::cout << all_time_blas_cf / NUM_QUERY_SETS << std::endl;
    // std::cout << all_time_eigen_cf / NUM_QUERY_SETS << std::endl;
    // return 0;

    // std::cout << "begin test" << std::endl;

    // double all_time_random = 0.0f;
    // double all_time_sequence = 0.0f;
    // // std::vector<float> res;
    // std::vector<float> test2_query_cluster_scores(262144 * 32);
    // std::vector<std::vector<float>> res1(100);
    // std::vector<std::vector<float>> res2(100);
    // for (int i = 0; i < NUM_QUERY_SETS; ++i) {
    //     for (int j = 0; j < 20000; j ++) {
    //         int num = dist_int(gen_int);
    //         random_numbers[j] = num;
    //     } 
    //     hnswlib::fast_dot_product_blas(262144, 128, 32, center_data.data(), (&query[i])->data, test_query_cluster_scores.data());  
    //     vectorset query_cluster = vectorset(test_query_cluster_scores.data(), nullptr, 262144, 32);
    //     hnswlib::fast_dot_product_blas(32, 128, 262144, (&query[i])->data, center_data.data(), test2_query_cluster_scores.data());  
    //     vectorset query_cluster2 = vectorset(test2_query_cluster_scores.data(), nullptr, 262144, 32);
    //     double t1_sequence = omp_get_wtime();
    //     res1[i].resize(20000);
    //     for (int j = 0; j < 20000; j++) {
    //         // float d0 = hnswlib::L2SqrCluster4Search(&query_center[i], &base[j], 0);
    //         float d1 = hnswlib::L2SqrCluster4Search(&query_cluster, &base[j], 0);
    //         // float d2 = hnswlib::L2SqrClusterAVX4Search(&query_cluster2, &base[j], 0);
    //         // std::cout << std::endl;
    //         // std::cout << d1 << " " << d2 << std::endl;
    //         res1[i][j] = d1;
    //     }
    //     // continue;
    //     double t2_sequence = omp_get_wtime();
    //     double t1_random = omp_get_wtime();
    //     res2[i].resize(20000);
    //     for (int j = 0; j < 20000; j++) {
    //         float d = hnswlib::L2SqrClusterAVX4Search(&query_cluster2, &base[j], 0);
    //         res2[i][j] = (d);
    //     }
    //     double t2_random = omp_get_wtime();
    //     std::cout << i << ": sequence 2w:  " << t2_sequence - t1_sequence << " random 2w: " << t2_random - t1_random<< std::endl;
    //     all_time_random += t2_random - t1_random;
    //     all_time_sequence += t2_sequence - t1_sequence;
    // }
    // std::cout << res1.size() << std::endl;
    // std::cout << res2.size() << std::endl;
    // // double t2 = omp_get_wtime();
    // std::cout << all_time_sequence / NUM_QUERY_SETS << std::endl;
    // std::cout << all_time_random / NUM_QUERY_SETS << std::endl;
    // return 0;

    if (!load_bf_from_cache) {
        GroundTruth ground_truth;
        ground_truth.build(VECTOR_DIM, base);
        std::cout<< "Generate BF Groundtruth" <<std::endl;
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < NUM_QUERY_SETS; ++i) {
            // std::vector<std::pair<int, float>> ground_truth_indices;
            // std::cout << i << std::endl;
            ground_truth.searchCluster(query[i], query_center[i], 4000, bf_ground_truth[i]);
        }
        std::cout<< "Generate BF Groundtruth Finish!" <<std::endl;
        std::ofstream outFile(ground_truth_file);
        for (int i = 0; i < NUM_QUERY_SETS; ++i) {
            for (int j = 0; j < 4000; j++) {
                outFile << bf_ground_truth[i][j].first << " " << bf_ground_truth[i][j].second << " ";
            }
            outFile << "\n";
            outFile.flush();
        }
        std::cout<< "write file BF Groundtruth Finish!" <<std::endl;
    }
    else {
        readGroundTruth(ground_truth_file, bf_ground_truth);
        readGroundTruth(ground_truth_file, bf_ground_truth_cf);
        std::cout<< "load BF Groundtruth Finish!" <<std::endl;
    }
    // for (int i = 0; i < 100; i++) {
    //     std::cout << i << " " << hnswlib::L2SqrVecCF(&query[i], &base[i], 0) << std::endl;
    // }
    // return 0;
    // for (int i = 0; i < 10000; i++) {
    //     // float chamfer_dist = hnswlib::L2SqrVecSet(&base[i], &base[i + 1], 0);
    //     float emd_dist = hnswlib::L2SqrVecEMD(&base[i], &base[i + 1], 0);
    //     float emd2_dist = hnswlib::L2SqrVecEMDBlas(&base[i], &base[i + 1], 0);
    //     std::cout << i << ' ' << emd_dist << ' ' << emd2_dist << std::endl;
    // }
    // return 0;


    // for (int i = 0; i < 32; i ++) {
    //     for (int j = 0; j < 500; j ++) {
    //         std::cout << i << " " << j << " " << query_cluster_scores[9 * 500 * 32 + i * 500 + j] << std::endl;
    //     }
    // }
    // std::cout << hnswlib::L2SqrCluster4Search(&query_center[9], &base[3338], 0) << std::endl;
    // return 0;
    index_file = "/home/zhoujin/project/forremove/VecSetSearch/hnswlib/examples/256clusterIndexTFIDF/";
    // index_file = "/home/zhoujin/project/forremove/VecSetSearch/hnswlib/examples/clusterFilterIndex/";
    // index_file = "/home/zhoujin/project/forremove/VecSetSearch/hnswlib/examples/clusterIndex/";
    // std::vector<int> temp_cluster_id(2761);

    Solution solution;
    if (rebuild) {
        std::vector<float>cluster_distance((long long) NUM_CLUSTER * NUM_CLUSTER);
        hnswlib::fast_dot_product_blas(NUM_CLUSTER, 128, NUM_CLUSTER, center_data.data(), center_data.data(), cluster_distance.data()); 
        solution.build(VECTOR_DIM, base, cluster_set, temp_cluster_id, cluster_distance);
        solution.save(index_file);
    } else {
        solution.load(index_file, VECTOR_DIM, base, cluster_set, temp_cluster_id);
    }

    for (int tmpef = 200; tmpef <= 2000; tmpef += 100) {
        double total_recall = 0.0;
        double total_cf_recall = 0.0;
        double total_dataset_hnsw_recall = 0.0;
        double total_wcf_bf_recall = 0.0;
        double total_cf_bf_recall = 0.0;
        double total_query_time = 0.0;
        double total_enrty_recall_10 = 0.0;
        double total_enrty_recall_30 = 0.0;
        double total_enrty_recall_50 = 0.0;
        double total_enrty_recall_100 = 0.0;

        l2_sqr_call_count.store(0);
        std::cout<<"Processing Queries HNSW"<<std::endl;
        // #pragma omp parallel for schedule(dynamic)
        // for (int tmpi = 0; tmpi < query182.size(); tmpi ++) {
        //     int i = query182[tmpi];
        //     std::cout << i << std::endl;
        for (int i = 0; i < NUM_QUERY_SETS; ++i) {
            // std::cout << i << std::endl;
            double wcf_bf_recall = calculate_recall_for_msmacro(bf_ground_truth[i], qrels[i]);
            // std::cout << i << std::endl;
            total_wcf_bf_recall += wcf_bf_recall;
            double cf_bf_recall = calculate_recall_for_msmacro(bf_ground_truth_cf[i], qrels[i]);
            // std::cout << i << std::endl;
            total_cf_bf_recall += cf_bf_recall;
            std::vector<std::pair<int, float>> solution_indices;
            // std::vector<hnswlib::labeltype> entry_points;
            // entry_points.resize(multi_entries_num);
            // std::cout << i << std::endl;
            try {
                // double query_time = solution.search_with_cluster(query[i], test_query_cluster_scores, center_data, K, tmpef, solution_indices);
                // std::cout << query[i].dim << " " <<  *(query[i].data) << " " << *(query[i].data + 32 * 128 - 1) << std::endl;
                // std::cout << test_query_cluster_scores[262144 * 32 - 1] << " " << col_query_cluster_scores[262144 * 32 - 1] << std::endl;
                // std::cout << solution_indices.size() << std::endl;
                // std::cout << center_data.size() << std::endl;
                // std::cout << graph_center_data.size() << std::endl;
                double query_time = solution.search_with_cluster(query[i], test_query_cluster_scores, col_query_cluster_scores, center_data, graph_center_data, K, tmpef, solution_indices);     
                total_query_time += query_time;
                double recall = calculate_recall(solution_indices, bf_ground_truth[i]);
                total_recall += recall;
                double cf_recall = calculate_recall(solution_indices, bf_ground_truth_cf[i]);
                total_cf_recall += cf_recall;
                double dataset_hnsw_recall = calculate_recall_for_msmacro(solution_indices, qrels[i]);
                total_dataset_hnsw_recall += dataset_hnsw_recall;
                std::cout << "Recall for query set " << i << ": " << dataset_hnsw_recall << " | " << recall << " " << wcf_bf_recall << " | " << cf_recall << " " << cf_bf_recall << " " << query_time << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Exception in search_with_cluster: " << e.what() << std::endl;
                exit(1);
            }
        }
        std::cout << "ef: " << tmpef << std::endl;
        std::cout << "Average Weighted CF BF Recall v.s. dataset label: " << total_wcf_bf_recall/NUM_QUERY_SETS<< std::endl;
        std::cout << "Average CF BF Recall v.s. dataset label: " << total_cf_bf_recall/NUM_QUERY_SETS << std::endl;
        std::cout << "Average our method Recall v.s. Weighted CF Brute Force: " << total_recall/NUM_QUERY_SETS << std::endl;
        std::cout << "Average our method Recall v.s. CF Brute Force: " << total_cf_recall/NUM_QUERY_SETS << std::endl;
        std::cout << "Average our method recall v.s. dataset label: " << total_dataset_hnsw_recall / NUM_QUERY_SETS << std::endl;
        std::cout << "Average query time: " << total_query_time/NUM_QUERY_SETS << " seconds" << std::endl;
        std::cout << "Average L2Sqr was called " << l2_sqr_call_count.load() / NUM_QUERY_SETS << " times." << std::endl;
    }
    return 0;

}
