#pragma once
#include "hnswlib.h"
#include "vectorset.h"
#include<algorithm>
#include <vector>
#include <cmath>
#include <omp.h> 
#include <Eigen/Dense>
using namespace Eigen;

namespace hnswlib {

static float
L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

static float
MyInnerProduct(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += (*pVect1) * (*pVect2); // 计算内积
        pVect1++;
        pVect2++;
    }
    return res;
}

#if defined(USE_AVX512)

// Favor using AVX512 if available.
static float
L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
            TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
            TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}
#endif

#if defined(USE_AVX)

// Favor using AVX if available.
static float
L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

#endif

#if defined(USE_SSE)

static float
L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC<float> L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

static float
L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}
#endif


#if defined(USE_SSE)
static float
L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);


    size_t qty4 = qty >> 2;

    const float *pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float
L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}
#endif

class L2VSSpace : public SpaceInterface<float>{
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
public:
    L2VSSpace(size_t dim){
        fstdistfunc_ = L2Sqr;
        dim_ = dim;
        data_size_ = sizeof(float*) + sizeof(int) * 2;
    }

    size_t get_data_size(){
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2VSSpace(){}
};

class L2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2Space(size_t dim) {
        fstdistfunc_ = L2Sqr;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
        else if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #elif defined(USE_AVX)
        if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = L2SqrSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = L2SqrSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = L2SqrSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2Space() {}
};

static int
L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    int res = 0;
    unsigned char *a = (unsigned char *) pVect1;
    unsigned char *b = (unsigned char *) pVect2;

    qty = qty >> 2;
    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    int res = 0;
    unsigned char* a = (unsigned char*)pVect1;
    unsigned char* b = (unsigned char*)pVect2;

    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

// SumMax L2
// static float
// L2SqrVecSet(const vectorset* q, const vectorset* p){
//     float sum = 0.0f;
//     // Iterate over each vector in q
//     for (size_t i = 0; i < q->vecnum; ++i) {
//         const float* vec_q = (q->data) + i * (q->dim); // Pointer to the i-th vector in q
//         float maxDist = 99999.9f; // Start with 0 to find maximum distance
//         for (size_t j = 0; j < p->vecnum; ++j) {
//             const float* vec_p = p->data + j * (p->dim); // Pointer to the j-th vector in p
//             maxDist = std::min(maxDist, L2Sqr(vec_q, vec_p, &((p->dim))));
//         }
//         sum += maxDist;
//     }
//     return sum / (q->vecnum);
// }

//SumMax + SumMax L2
static float L2SqrVecSet(const vectorset* q, const vectorset* p) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;

    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = L2Sqr;
    #endif

    //#pragma omp parallel for num_threads(4) reduction(+:sum1)
    #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < q->vecnum; ++i) {
        const float* vec_q = q->data + i * q->dim;
        float maxDist = 99999.9f;
        for (size_t j = 0; j < p->vecnum; ++j) {
            const float* vec_p = p->data + j * p->dim;
            float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
            maxDist = std::min(maxDist, dist);
        }
        sum1 += maxDist;
    }


    //#pragma omp parallel for num_threads(4) reduction(+:sum2)
    #pragma omp simd reduction(+:sum2)
    for (size_t i = 0; i < p->vecnum; ++i) {
        const float* vec_p = p->data + i * p->dim;
        float maxDist = 99999.9f;
        for (size_t j = 0; j < q->vecnum; ++j) {
            const float* vec_q = q->data + j * q->dim;
            float dist = L2Sqrfunc_(vec_p, vec_q, &q->dim);
            maxDist = std::min(maxDist, dist);
        }
        sum2 += maxDist;
    }

    return (sum1 + sum2) / (q->vecnum + p->vecnum);
}

// Hausdorff Distance L2
// static float
// L2SqrVecSet(const vectorset* q, const vectorset* p){
//     float sum1 = 0.0f;
//     // Iterate over each vector in q
//     for (size_t i = 0; i < q->vecnum; ++i) {
//         const float* vec_q = (q->data) + i * (q->dim); // Pointer to the i-th vector in q
//         float maxDist = 99999.9f;
//         for (size_t j = 0; j < p->vecnum; ++j) {
//             const float* vec_p = p->data + j * (p->dim); // Pointer to the j-th vector in p
//             maxDist = std::min(maxDist, L2Sqr(vec_q, vec_p, &((p->dim))));
//         }
//         sum1 = std::max(sum1, maxDist);
//     }
//     float sum2 = 0.0f;
//     // Iterate over each vector in p
//     for (size_t i = 0; i < p->vecnum; ++i) {
//         const float* vec_p = (p->data) + i * (p->dim); // Pointer to the i-th vector in p
//         float maxDist = 99999.9f; // Start with 0 to find maximum distance
//         for (size_t j = 0; j < q->vecnum; ++j) {
//             const float* vec_q = q->data + j * (q->dim); // Pointer to the j-th vector in q
//             maxDist = std::min(maxDist, L2Sqr(vec_p, vec_q, &((q->dim))));
//         }
//         sum2 = std::max(sum2, maxDist);
//     }
//     return  std::max(sum1, sum2);
// }

// SumSum L2
// static float
// L2SqrVecSet(const vectorset* q, const vectorset* p){
//     float sum = 0.0f;
//     // Iterate over each vector in q
//     for (size_t i = 0; i < q->vecnum; ++i) {
//         const float* vec_q = (q->data) + i * (q->dim); // Pointer to the i-th vector in q
//         for (size_t j = 0; j < p->vecnum; ++j) {
//             const float* vec_p = p->data + j * (p->dim); // Pointer to the j-th vector in p
//             sum += L2Sqr(vec_q, vec_p, &((p->dim)));
//         }
//     }
//     return sum / (q->vecnum + p->vecnum);
// }

// MinMin L2
// static float
// L2SqrVecSet(const vectorset* q, const vectorset* p){
//     float sum = 99999.9f;
//     // Iterate over each vector in q
//     for (size_t i = 0; i < q->vecnum; ++i) {
//         const float* vec_q = (q->data) + i * (q->dim); // Pointer to the i-th vector in q
//         for (size_t j = 0; j < p->vecnum; ++j) {
//             const float* vec_p = p->data + j * (p->dim); // Pointer to the j-th vector in p
//             sum = std::min(sum, L2Sqr(vec_q, vec_p, &((p->dim))));
//         }
//     }
//     return sum;
// }

// MaxMax L2
// static float
// L2SqrVecSet(const vectorset* q, const vectorset* p){
//     float sum = 0.0f;
//     // Iterate over each vector in q
//     for (size_t i = 0; i < q->vecnum; ++i) {
//         const float* vec_q = (q->data) + i * (q->dim); // Pointer to the i-th vector in q
//         for (size_t j = 0; j < p->vecnum; ++j) {
//             const float* vec_p = p->data + j * (p->dim); // Pointer to the j-th vector in p
//             sum = std::max(sum, L2Sqr(vec_q, vec_p, &((p->dim))));
//         }
//     }
//     return sum;
// }

// regularized_chamfer_distance
// static float 
// L2SqrVecSet(const vectorset* q, const vectorset* p){
//     double lambda_reg = 1e-6;
//     int max_iter = 100;
//     double tol = 1e-6;
//     MatrixXd cost_matrix_AB = MatrixXd::Zero(q->vecnum, p->vecnum);
//     for (size_t i = 0; i < q->vecnum; ++i) {
//         const float* vec_q = (q->data) + i * (q->dim); // Pointer to the i-th vector in q
//         for (size_t j = 0; j < p->vecnum; ++j) {
//             const float* vec_p = p->data + j * (p->dim); // Pointer to the j-th vector in p
//             cost_matrix_AB(i, j) = L2Sqr(vec_q, vec_p, &((p->dim)));
//         }
//     }
//     MatrixXd cost_matrix_BA = cost_matrix_AB.transpose();

//     // Initialize the dual variables
//     VectorXd u_AB = VectorXd::Ones(q->vecnum)/q->vecnum;
//     VectorXd v_AB = VectorXd::Ones(p->vecnum)/p->vecnum;
//     VectorXd u_BA = VectorXd::Ones(p->vecnum)/p->vecnum;
//     VectorXd v_BA = VectorXd::Ones(q->vecnum)/q->vecnum;

//     // Sinkhorn-Knopp algorithm for regularized transport
//     MatrixXd K_AB = (-lambda_reg * cost_matrix_AB).array().exp();
//     MatrixXd K_BA = (-lambda_reg * cost_matrix_BA).array().exp();

//     for (int iter = 0; iter < max_iter; ++iter) {
//         // A - > B updates
//         VectorXd u_AB_prev = u_AB;
//         u_AB = (K_AB* v_AB).cwiseInverse();
//         v_AB = (K_AB.transpose() * u_AB).cwiseInverse();

//         // B - > A updates
//         VectorXd u_BA_prev = u_BA;
//         u_BA = (K_BA * v_BA).cwiseInverse();
//         v_BA = (K_BA.transpose() * u_BA).cwiseInverse();

//         // Normalize u and v
//         u_AB /= u_AB.sum();
//         v_AB /= v_AB.sum();
//         u_BA /= u_BA.sum();
//         v_BA /= v_BA.sum();

//         // Check for convergence
//         double err_AB = (u_AB - u_AB_prev).norm();
//         double err_BA = (u_BA - u_BA_prev).norm();
//         if (err_AB < tol && err_BA < tol) {
//             break;
//         }
//     }
//     // Compute the transport plan
//     MatrixXd transport_plan_AB = u_AB.asDiagonal() * K_AB * v_AB.asDiagonal();
//     MatrixXd transport_plan_BA = u_BA.asDiagonal() * K_BA * v_BA.asDiagonal();

//     // Compute the Regularized Chamfer Distance
//     double regularized_chamfer_distance_AB = (transport_plan_AB.array() * cost_matrix_AB.array()).sum();
//     double regularized_chamfer_distance_BA = (transport_plan_BA.array() * cost_matrix_BA.array()).sum();
//     double symmetric_regularized_chamfer_distance = regularized_chamfer_distance_AB + regularized_chamfer_distance_BA;

//     return symmetric_regularized_chamfer_distance;
// }


// SumMax MIPS
// static float
// L2SqrVecSet(const vectorset* q, const vectorset* p){
//     float sum = 0.0f;
//     // Iterate over each vector in q
//     for (size_t i = 0; i < q->vecnum; ++i) {
//         float maxDist = -99999.9f; // Start with 0 to find maximum distance
//         const float* vec_q = (q->data) + i * (q->dim); // Pointer to the i-th vector in q
//         for (size_t j = 0; j < p->vecnum; ++j) {
//             const float* vec_p = p->data + j * (p->dim); // Pointer to the j-th vector in p
//             maxDist = std::max(maxDist, MyInnerProduct(vec_q, vec_p, &((p->dim))));
//         }
//         sum += maxDist;
//     }
//     return -sum / (q->vecnum);
// }


// MaxMax MIPS
// static float
// L2SqrVecSet(const vectorset* q, const vectorset* p){
//     float sum = -99999.9f;
//     // Iterate over each vector in q
//     for (size_t i = 0; i < q->vecnum; ++i) {
//         const float* vec_q = (q->data) + i * (q->dim); // Pointer to the i-th vector in q
//         for (size_t j = 0; j < p->vecnum; ++j) {
//             const float* vec_p = p->data + j * (p->dim); // Pointer to the j-th vector in p
//             sum = std::max(sum, MyInnerProduct(vec_q, vec_p, &((p->dim))));
//         }
//     }
//     return -sum;
// }

// Test single L2
// static float
// L2SqrVecSet(const vectorset* q, const vectorset* p){
//     float sum = 0.0f;
//     const float* vec_q = (q->data) ; // Pointer to the i-th vector in q
//     const float* vec_p = p->data; // Pointer to the j-th vector in p
//     sum =  L2Sqr(vec_q, vec_p, &((p->dim)));
//     return sum;
// }

// Test single MIPS
// static float
// L2SqrVecSet(const vectorset* q, const vectorset* p){
//     float sum = 0.0f;
//     const float* vec_q = (q->data) ; // Pointer to the i-th vector in q
//     const float* vec_p = p->data; // Pointer to the j-th vector in p
//     sum =  MyInnerProduct(vec_q, vec_p, &((p->dim)));
//     return - sum;
// }



class L2SpaceI : public SpaceInterface<int> {
    DISTFUNC<int> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2SpaceI(size_t dim) {
        if (dim % 4 == 0) {
            fstdistfunc_ = L2SqrI4x;
        } else {
            fstdistfunc_ = L2SqrI;
        }
        dim_ = dim;
        data_size_ = dim * sizeof(unsigned char);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<int> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2SpaceI() {}
};
}  // namespace hnswlib
