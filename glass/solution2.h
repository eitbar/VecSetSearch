#include <vector>
#include <chrono>
#include <iostream>
#include <utility>
#include <algorithm>

#include "./bindings.cpp"

const int NUM_THREADS = 16;

class Solution {
public:
    Solution() = default;

    ~Solution()
    {
        delete m_graph;
        delete m_searcher;
    }

    void build(int d, const std::vector<float> &base, const std::vector<float> &vec_num){
        dim = d;
        int rows = base.size() / d;
        std::unique_ptr<glass::FP32VCQuantizer<glass::Metric::L2, d>> buildQuant(new glass::FP32VCQuantizer<glass::Metric::L2, d>(d));
        buildQuant->train(base.data(), vec_num.data(), rows);

        std::unique_ptr<Index> hnsw_index(new IndexSQ8("HNSW", d, "L2", R, L));
        m_graph = new Graph(hnsw_index->build((uint8_t *)buildQuant->get_data(0), rows));

        m_searcher = new Searcher(*m_graph, base.data(), rows, d, "L2", )
    }

    void search(const std::vector<float> & query, const int num_vec, const int k, int* res){

    }
private:
    Graph *m_graph;
    Searcher *m_searcher;
    int R;
    int L;
    int EF;
    int LEVEL;
}