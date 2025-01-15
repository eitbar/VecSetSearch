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
constexpr int MSMACRO_TEST_NUMBER = 354;
constexpr int NUM_BASE_SETS = 25000 * MSMACRO_TEST_NUMBER;
constexpr int NUM_QUERY_SETS = 100;
constexpr int QUERY_VECTOR_COUNT = 32;
constexpr int K = 100;

class GroundTruth {
public:
    void build(int d, const std::vector<vectorset>& base) {
        dimension = d;
        base_vectors = base;
    }

    void search(const vectorset query, int k, std::vector<std::pair<int, float>>& res) const {
        // res.clear();
        std::vector<std::pair<float, int>> distances;

        // Calculate Chamfer distance between query set and each base set
        //std::cout<<"Calc Dis"<<std::endl;
        int base_offset = 0;
        for (size_t i = 0; i < base_vectors.size(); ++i) {
            if (i % 1000000 == 0) {
                std::cout << "calc distance for " << i << std::endl;
            }
            float chamfer_dist = hnswlib::L2SqrVecSet(&query, &base_vectors[i], 0);
            distances.push_back({chamfer_dist, static_cast<int>(i)});
        }
        //std::cout<<"return ans"<<std::endl;
        // Sort distances to find top-k nearest neighbors
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        for (int i = 0; i < k; ++i) {
            res[i] = std::make_pair(distances[i].second, distances[i].first);
        }
        
    }

private:
    int dimension;
    std::vector<vectorset> base_vectors;
};

class Solution {
public:
    void build(int d, const std::vector<vectorset>& base) {
        double time = omp_get_wtime();
        dimension = d;
        base_vectors = base;
        space_ptr = new hnswlib::L2VSSpace(dimension);
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(space_ptr, base_vectors.size() + 1, 16, 80);
        #pragma omp parallel for schedule(dynamic)
        for(hnswlib::labeltype i = 0; i < base_vectors.size(); i++){
            // std::cout << i << std::endl;
            if (i % 1000 == 0) {
                std::cout << i << std::endl;
            }
            alg_hnsw->addPoint(&base_vectors[i], i);            
        }
        alg_hnsw->setEf(200);
        std::cout << "Build time: " << omp_get_wtime() - time << "sec"<<std::endl;
        // Add any necessary pre-computation or indexing for the optimized search
    }

    double search(const vectorset query, int k, int ef, std::vector<std::pair<int, float>>& res) const {
        res.clear();
        alg_hnsw->setEf(ef);
        double start_time = omp_get_wtime();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnFineEdge(&query, k);
        for(int i = 0; i < k; i++){
            res.push_back(std::make_pair(result.top().second, result.top().first));
            result.pop();
        }
        double end_time = omp_get_wtime();
        return end_time - start_time;
    }

    double searchFromEntries(const vectorset query, int k, int ef, std::vector<hnswlib::labeltype>& entry_points, std::vector<std::pair<int, float>>& res) const {
        res.clear();
        alg_hnsw->setEf(ef);
        double start_time = omp_get_wtime();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnParaFromEntries(&query, k, entry_points);
        for(int i = 0; i < k; i++){
            res.push_back(std::make_pair(result.top().second, result.top().first));
            result.pop();
        }
        double end_time = omp_get_wtime();
        return end_time - start_time;
    }

    void save(const std::string &location) {
        double time = omp_get_wtime();
        alg_hnsw->saveIndex(location);
        std::cout << "save time: " << omp_get_wtime() - time << "sec"<<std::endl;
    }

    void load(const std::string &location, int d, const std::vector<vectorset>& base) {
        double time = omp_get_wtime();
        space_ptr = new hnswlib::L2VSSpace(d);
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(space_ptr, base.size() + 1, 16, 80);
        alg_hnsw->loadIndex(location, space_ptr);
        // #pragma omp parallel for schedule(dynamic)
        for(hnswlib::labeltype i = 0; i < base.size(); i++){
            alg_hnsw->loadDataAddress(&base[i], i);            
        }
        std::cout << "load time: " << omp_get_wtime() - time << "sec"<<std::endl;
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
                       int file_numbers, std::vector<std::vector<int>>& qrels) {
    long long offset = 0;  
    long long all_elements = 0;   
    std::string qembfile_name = "/home/zhoujin/vecDB_publi_data/0.6b_128d_dataset/qembs_32_6980.npy";
    std::string qrelfile_name = "/home/zhoujin/vecDB_publi_data/0.6b_128d_dataset/qrels_6980.tsv";    

    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = "/home/zhoujin/vecDB_publi_data/0.6b_128d_dataset/encoding" + std::to_string(i) + "_float16.npy";
        std::string lensfile_name = "/home/zhoujin/vecDB_publi_data/0.6b_128d_dataset/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];
        // int* lens_data = lens_npy.data<int>();
        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = lens_npy.shape[0];
        std::cout << "Processing file " << i << std::endl;
        // assert (doc_num == 25000);
        
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
    size_t num_qembs_elements = NUM_QUERY_SETS * qembs_npy.shape[1] * qembs_npy.shape[2];
    size_t q_num = NUM_QUERY_SETS;

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
        if (solution_set.size() >= K) {
            break;
        }
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
        if (solution_set.size() >= K) {
            break;
        }
    }
    // std::cout << std::endl;

    for (const auto& pair : ground_truth_indices) {
        ground_truth_set.insert(pair.first);
        // std::cout << pair.first << " ";
        if (ground_truth_set.size() >= K) {
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
    std::vector<std::vector<int>> qrels;
    std::vector<std::vector<std::pair<int, float>>> bf_ground_truth(
        NUM_QUERY_SETS, std::vector<std::pair<int, float>>(K, {0, 0.0f})
    );
    bool test_subset = true;
    bool load_bf_from_cache = true;
    bool rebuild = false;
    int dist_metric = 2;
    int multi_entries_num = 40;
    int multi_entries_range = 100;
    std::mt19937 gen(42);                    // 使用Mersenne Twister引擎
    std::uniform_int_distribution<int> dist(1, std::numeric_limits<int>::max());
    std::string ground_truth_file, index_file;
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
            ground_truth_file = "../examples/caches/ground_truth_single_summax_l2_top100.txt";
            index_file = "../examples/localIndex/8m_single_summax_l2.bin";
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
    
    if (test_subset) {
        // test on collected 95k msmacro subset
        base_data.resize((long long) 96000 * VECTOR_DIM * 80);
        query_data.resize((long long) NUM_QUERY_SETS * VECTOR_DIM * 32);
        subset_test_msmarco(base_data, base, query_data, query, qrels);     
    } else {
        // test on all msmacro dataset
        base_data.resize((long long) 25000 * MSMACRO_TEST_NUMBER * 128 * 80);
        query_data.resize((long long) NUM_QUERY_SETS * 128 * 32 + 1);
        load_from_msmarco(base_data, base, query_data, query, MSMACRO_TEST_NUMBER, qrels);
    }

    if (!load_bf_from_cache) {
        GroundTruth ground_truth;
        ground_truth.build(VECTOR_DIM, base);
        std::cout<< "Generate BF Groundtruth" <<std::endl;
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < NUM_QUERY_SETS; ++i) {
            // std::vector<std::pair<int, float>> ground_truth_indices;
            std::cout << i << std::endl;
            ground_truth.search(query[i], K, bf_ground_truth[i]);
        }
        std::cout<< "Generate BF Groundtruth Finish!" <<std::endl;
        std::ofstream outFile(ground_truth_file);
        for (int i = 0; i < NUM_QUERY_SETS; ++i) {
            for (int j = 0; j < K; j++) {
                outFile << bf_ground_truth[i][j].first << " " << bf_ground_truth[i][j].second << " ";
            }
            outFile << "\n";
            outFile.flush();
        }
        std::cout<< "write file BF Groundtruth Finish!" <<std::endl;
    }
    else {
        readGroundTruth(ground_truth_file, bf_ground_truth);
    }

    Solution solution;
    if (rebuild) {
        solution.build(VECTOR_DIM, base);
        solution.save(index_file);
    } else {
        solution.load(index_file, VECTOR_DIM, base);
    }

    for (int tmpef = 100; tmpef <= 100; tmpef += 100) {
        double total_recall = 0.0;
        double total_dataset_hnsw_recall = 0.0;
        double total_dataset_bf_recall = 0.0;
        double total_query_time = 0.0;
        double total_enrty_recall_10 = 0.0;
        double total_enrty_recall_30 = 0.0;
        double total_enrty_recall_50 = 0.0;
        double total_enrty_recall_100 = 0.0;


        std::cout<<"Processing Queries HNSW"<<std::endl;
        // #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < NUM_QUERY_SETS; ++i) {
            std::vector<std::pair<int, float>> solution_indices;
            std::vector<hnswlib::labeltype> entry_points;
            entry_points.resize(multi_entries_num);
            // std::vector<int> entry_point_index = generateEntriesIndex(multi_entries_num, multi_entries_range);
            // std::unordered_set<int> ground_truth_set;
            // for (const auto& pair : bf_ground_truth[i]) {
            //     ground_truth_set.insert(pair.first);
            //     if (ground_truth_set.size() >= 50) {
            //         break;
            //     }
            // }
            // for (int j = 0; j < multi_entries_num; j++) {
            //     entry_points[j] = bf_ground_truth[i][entry_point_index[j]].first;
            // }
            for (int j = 0; j < multi_entries_num; j++) {
                int tmp_ind = dist(gen) % base.size();
                // while (ground_truth_set.find(tmp_ind) != ground_truth_set.end()) {
                //     tmp_ind = dist(gen) % base.size();
                // }
                entry_points[j] = tmp_ind;
            }
            // std::cout << bf_ground_truth[i].size() << std::endl;
            // std::cout<<entry_points.size()<<std::endl;
            // double query_time = solution.search(query[i], K, solution_indices);
            // double query_time = solution.searchFromEntries(query[i], K, tmpef, entry_points, solution_indices);
            double query_time = solution.search(query[i], K, tmpef, solution_indices);
            // for (int j = 0; j < K; j ++) {
            //     std::cout << solution_indices[j].first << " " << solution_indices[j].second << " ";
            // }
            // std::cout << std::endl;

            // for (int j = 0; j < K; j ++) {
            //     std::cout << bf_ground_truth[i][j].first << " " << bf_ground_truth[i][j].second << " ";
            // }
            // std::cout << std::endl;        
            total_query_time += query_time;
            double recall = calculate_recall(solution_indices, bf_ground_truth[i]);
            total_recall += recall;
            double entry_recall = calculate_entry_recall(entry_points, bf_ground_truth[i], 10);
            total_enrty_recall_10 += entry_recall;
            // if (entry_recall > 0) {
            //     for (int j = 0; j < 80;j++) {
            //         std::cout << entry_points[j] << std::endl;
            //     }
            //     std::cout << "===" << std::endl;
            //     for (int j = 0; j < 10; j++) {
            //         std::cout << bf_ground_truth[i][j].first << std::endl;
            //     }
            //     break;
            // }
            entry_recall = calculate_entry_recall(entry_points, bf_ground_truth[i], 30);
            total_enrty_recall_30 += entry_recall;
            entry_recall = calculate_entry_recall(entry_points, bf_ground_truth[i], 50);
            total_enrty_recall_50 += entry_recall;
            entry_recall = calculate_entry_recall(entry_points, bf_ground_truth[i], 100);
            total_enrty_recall_100 += entry_recall;
            double dataset_hnsw_recall = calculate_recall_for_msmacro(solution_indices, qrels[i]);
            total_dataset_hnsw_recall += dataset_hnsw_recall;
            double dataset_bf_recall = calculate_recall_for_msmacro(bf_ground_truth[i], qrels[i]);
            total_dataset_bf_recall += dataset_bf_recall;
            std::cout << "Recall for query set " << i << ": " << recall << " " << dataset_hnsw_recall << " " << dataset_bf_recall << std::endl;
        }

        // Calculate average recall
        double average_recall = total_recall / NUM_QUERY_SETS;
        double average_dataset_hnsw_recall = total_dataset_hnsw_recall / NUM_QUERY_SETS;
        double average_dataset_bf_recall = total_dataset_bf_recall / NUM_QUERY_SETS;
        std::cout << "ef: " << tmpef << std::endl;
        std::cout << "Average Recall: " << average_recall << std::endl;
        std::cout << "Average Dataset HNSW Recall: " << average_dataset_hnsw_recall << std::endl;
        std::cout << "Average Dataset BF Recall: " << average_dataset_bf_recall << std::endl;
        std::cout << "Average entry recall: " << total_enrty_recall_10/NUM_QUERY_SETS << " " << total_enrty_recall_30/NUM_QUERY_SETS << " " << total_enrty_recall_50/NUM_QUERY_SETS << " " << total_enrty_recall_100/NUM_QUERY_SETS << std::endl;
        std::cout << "Average query time: " << total_query_time/NUM_QUERY_SETS << " seconds" << std::endl;
        std::cout << "Average L2Sqr was called " << l2_sqr_call_count.load() / NUM_QUERY_SETS << " times." << std::endl;
        std::cout << "Average L2Vec was called " << l2_vec_call_count.load() / NUM_QUERY_SETS << " times." << std::endl;
    }
    return 0;

}
