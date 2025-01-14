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
// static float L2SqrVecSet(const vectorset* q, const vectorset* p, int level) {
//     float sum1 = 0.0f;
//     float sum2 = 0.0f;
//     // level = 0;
//     float (*L2Sqrfunc_)(const void*, const void*, const void*);
//     #if defined(USE_AVX512)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
//     #elif defined(USE_AVX)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
//     #else 
//     L2Sqrfunc_ = L2Sqr;
//     #endif

//     // 根据 level 参数选择不同数量的向量进行计算，并确保至少为 1
//     size_t q_vecnum = std::max(static_cast<size_t>(1), q->vecnum / (1 << level));
//     size_t p_vecnum = std::max(static_cast<size_t>(1), p->vecnum / (1 << level));

//     //#pragma omp parallel for num_threads(4) reduction(+:sum1)
//     #pragma omp simd reduction(+:sum1)
//     for (size_t i = 0; i < q_vecnum; ++i) {
//         const float* vec_q = q->data + i * (level + 1) * q->dim;
//         float maxDist = 99999.9f;
//         for (size_t j = 0; j < p_vecnum; ++j) {
//             const float* vec_p = p->data + j * p->dim;
//             float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
//             maxDist = std::min(maxDist, dist);
//         }
//         sum1 += maxDist;
//     }


//     //#pragma omp parallel for num_threads(4) reduction(+:sum2)
//     #pragma omp simd reduction(+:sum2)
//     for (size_t i = 0; i < p_vecnum; ++i) {
//         const float* vec_p = p->data + i * p->dim;
//         float maxDist = 99999.9f;
//         for (size_t j = 0; j < q_vecnum; ++j) {
//             const float* vec_q = q->data + j * q->dim;
//             float dist = L2Sqrfunc_(vec_p, vec_q, &q->dim);
//             maxDist = std::min(maxDist, dist);
//         }
//         sum2 += maxDist;
//     }

//     return (sum1 + sum2) / (q_vecnum + p_vecnum);
// }


// static float L2SqrVecSet4Search(const vectorset* q, const vectorset* p, int level) {
//     float sum1 = 0.0f;
//     float sum2 = 0.0f;
//     // level = 0;
//     float (*L2Sqrfunc_)(const void*, const void*, const void*);
//     #if defined(USE_AVX512)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
//     #elif defined(USE_AVX)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
//     #else 
//     L2Sqrfunc_ = L2Sqr;
//     #endif

//     // 根据 level 参数选择不同数量的向量进行计算，并确保至少为 1
//     size_t p_vecnum = std::max(static_cast<size_t>(1), p->vecnum / (1 << level));

//     //#pragma omp parallel for num_threads(4) reduction(+:sum1)
//     #pragma omp simd reduction(+:sum1)
//     for (size_t i = 0; i < q->vecnum; ++i) {
//         const float* vec_q = q->data + i * q->dim;
//         float maxDist = 99999.9f;
//         for (size_t j = 0; j < p_vecnum; ++j) {
//             const float* vec_p = p->data + j * p->dim;
//             float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
//             maxDist = std::min(maxDist, dist);
//         }
//         sum1 += maxDist;
//     }


//     //#pragma omp parallel for num_threads(4) reduction(+:sum2)
//     #pragma omp simd reduction(+:sum2)
//     for (size_t i = 0; i < p_vecnum; ++i) {
//         const float* vec_p = p->data + i * p->dim;
//         float maxDist = 99999.9f;
//         for (size_t j = 0; j < q->vecnum; ++j) {
//             const float* vec_q = q->data + j * q->dim;
//             float dist = L2Sqrfunc_(vec_p, vec_q, &q->dim);
//             maxDist = std::min(maxDist, dist);
//         }
//         sum2 += maxDist;
//     }

//     return (sum1 + sum2) / (q->vecnum + p_vecnum);
// }


//SumMax L2
// static float L2SqrVecSet(const vectorset* q, const vectorset* p, int level) {
//     float sum1 = 0.0f;
//     // level = 0;
//     float (*L2Sqrfunc_)(const void*, const void*, const void*);
//     #if defined(USE_AVX512)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
//     #elif defined(USE_AVX)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
//     #else 
//     L2Sqrfunc_ = L2Sqr;
//     #endif

//     // 根据 level 参数选择不同数量的向量进行计算，并确保至少为 1
//     size_t q_vecnum = std::max(static_cast<size_t>(1), q->vecnum / (1 << level));
//     size_t p_vecnum = std::max(static_cast<size_t>(1), p->vecnum / (1 << level));

//     //#pragma omp parallel for num_threads(4) reduction(+:sum1)
//     #pragma omp simd reduction(+:sum1)
//     for (size_t i = 0; i < q_vecnum; ++i) {
//         const float* vec_q = q->data + i * (level + 1) * q->dim;
//         float maxDist = 99999.9f;
//         for (size_t j = 0; j < p_vecnum; ++j) {
//             const float* vec_p = p->data + j * p->dim;
//             float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
//             maxDist = std::min(maxDist, dist);
//         }
//         sum1 += maxDist;
//     }

//     return sum1;
// }


// static float L2SqrVecSet4Search(const vectorset* q, const vectorset* p, int level) {
//     float sum1 = 0.0f;
//     // level = 0;
//     float (*L2Sqrfunc_)(const void*, const void*, const void*);
//     #if defined(USE_AVX512)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
//     #elif defined(USE_AVX)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
//     #else 
//     L2Sqrfunc_ = L2Sqr;
//     #endif

//     // 根据 level 参数选择不同数量的向量进行计算，并确保至少为 1
//     size_t p_vecnum = std::max(static_cast<size_t>(1), p->vecnum / (1 << level));

//     //#pragma omp parallel for num_threads(4) reduction(+:sum1)
//     #pragma omp simd reduction(+:sum1)
//     for (size_t i = 0; i < q->vecnum; ++i) {
//         const float* vec_q = q->data + i * q->dim;
//         float maxDist = 99999.9f;
//         for (size_t j = 0; j < p_vecnum; ++j) {
//             const float* vec_p = p->data + j * p->dim;
//             float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
//             maxDist = std::min(maxDist, dist);
//         }
//         sum1 += maxDist;
//     }

//     return sum1;
// }

// new bi SumMax L2
// static float L2SqrVecSet(const vectorset* q, const vectorset* p, int level) {
//     float sum1 = 0.0f;
//     float sum2 = 0.0f;
//     // level = 0;
//     float (*L2Sqrfunc_)(const void*, const void*, const void*);
//     #if defined(USE_AVX512)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
//     #elif defined(USE_AVX)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
//     #else 
//     L2Sqrfunc_ = L2Sqr;
//     #endif

//     // 根据 level 参数选择不同数量的向量进行计算，并确保至少为 1
//     size_t q_vecnum = std::max(static_cast<size_t>(1), q->vecnum / (1 << level));
//     size_t p_vecnum = std::max(static_cast<size_t>(1), p->vecnum / (1 << level));

//     //#pragma omp parallel for num_threads(4) reduction(+:sum1)
//     #pragma omp simd reduction(+:sum1)
//     for (size_t i = 0; i < q_vecnum; ++i) {
//         const float* vec_q = q->data + i * (level + 1) * q->dim;
//         float maxDist = 99999.9f;
//         for (size_t j = 0; j < p_vecnum; ++j) {
//             const float* vec_p = p->data + j * p->dim;
//             float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
//             maxDist = std::min(maxDist, dist);
//         }
//         sum1 += maxDist;
//     }


//     //#pragma omp parallel for num_threads(4) reduction(+:sum2)
//     #pragma omp simd reduction(+:sum2)
//     for (size_t i = 0; i < p_vecnum; ++i) {
//         const float* vec_p = p->data + i * p->dim;
//         float maxDist = 99999.9f;
//         for (size_t j = 0; j < q_vecnum; ++j) {
//             const float* vec_q = q->data + j * q->dim;
//             float dist = L2Sqrfunc_(vec_p, vec_q, &q->dim);
//             maxDist = std::min(maxDist, dist);
//         }
//         sum2 += maxDist;
//     }

//     return sum1 / q_vecnum + sum2 / p_vecnum;
// }


// static float L2SqrVecSet4Search(const vectorset* q, const vectorset* p, int level) {
//     float sum1 = 0.0f;
//     float sum2 = 0.0f;
//     // level = 0;
//     float (*L2Sqrfunc_)(const void*, const void*, const void*);
//     #if defined(USE_AVX512)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
//     #elif defined(USE_AVX)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
//     #else 
//     L2Sqrfunc_ = L2Sqr;
//     #endif

//     // 根据 level 参数选择不同数量的向量进行计算，并确保至少为 1
//     size_t p_vecnum = std::max(static_cast<size_t>(1), p->vecnum / (1 << level));

//     //#pragma omp parallel for num_threads(4) reduction(+:sum1)
//     #pragma omp simd reduction(+:sum1)
//     for (size_t i = 0; i < q->vecnum; ++i) {
//         const float* vec_q = q->data + i * q->dim;
//         float maxDist = 99999.9f;
//         for (size_t j = 0; j < p_vecnum; ++j) {
//             const float* vec_p = p->data + j * p->dim;
//             float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
//             maxDist = std::min(maxDist, dist);
//         }
//         sum1 += maxDist;
//     }

//     //#pragma omp parallel for num_threads(4) reduction(+:sum2)
//     #pragma omp simd reduction(+:sum2)
//     for (size_t i = 0; i < p_vecnum; ++i) {
//         const float* vec_p = p->data + i * p->dim;
//         float maxDist = 99999.9f;
//         for (size_t j = 0; j < q->vecnum; ++j) {
//             const float* vec_q = q->data + j * q->dim;
//             float dist = L2Sqrfunc_(vec_p, vec_q, &q->dim);
//             maxDist = std::min(maxDist, dist);
//         }
//         sum2 += maxDist;
//     }

//     return sum1 / q->vecnum + sum2 /  p_vecnum;
// }

// random 版本

static float L2SqrVecSet(const vectorset* q, const vectorset* p, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = L2Sqr;
    #endif
    size_t q_vecnum = std::max(static_cast<size_t>(1), q->vecnum / (1 << level));
    size_t p_vecnum = std::max(static_cast<size_t>(1), p->vecnum / (1 << level));
    // 使用随机数引擎打乱序列
    if (level != 0) { 
        std::vector<int> random_sequence_p, random_sequence_q;
        for (int i = 0; i < p->vecnum; ++i) {
            random_sequence_p.push_back(i); // 初始化为 1 到 n
        }
        for (int i = 0; i < q->vecnum; ++i) {
            random_sequence_q.push_back(i); // 初始化为 1 到 n
        }        
        std::random_device rd;  // 随机设备（种子）
        std::mt19937 gen(rd()); // 使用 mt19937 引擎
        std::shuffle(random_sequence_p.begin(), random_sequence_p.end(), gen);
        std::shuffle(random_sequence_q.begin(), random_sequence_q.end(), gen);
        std::vector<std::vector<float>> dist_matrix(q_vecnum, std::vector<float>(p_vecnum));
        //#pragma omp parallel for num_threads(4) reduction(+:sum1)
        #pragma omp simd reduction(+:sum1)
        for (size_t i = 0; i < q_vecnum; ++i) {
            const float* vec_q = q->data + random_sequence_q[i] * q->dim;
            float maxDist = 99999.9f;
            for (size_t j = 0; j < p_vecnum; ++j) {
                const float* vec_p = p->data + random_sequence_p[j] * p->dim;
                float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
                dist_matrix[i][j] = dist;
                maxDist = std::min(maxDist, dist);
            }
            sum1 += maxDist;
        }


        //#pragma omp parallel for num_threads(4) reduction(+:sum2)
        #pragma omp simd reduction(+:sum2)
        for (size_t i = 0; i < p_vecnum; ++i) {
            float maxDist = 99999.9f;
            for (size_t j = 0; j < q_vecnum; ++j) {
                float dist = dist_matrix[j][i];
                maxDist = std::min(maxDist, dist);
            }
            sum2 += maxDist;
        }
    } else {
        std::vector<std::vector<float>> dist_matrix(q_vecnum, std::vector<float>(p_vecnum));
        //#pragma omp parallel for num_threads(4) reduction(+:sum1)
        #pragma omp simd reduction(+:sum1)
        for (size_t i = 0; i < q_vecnum; ++i) {
            const float* vec_q = q->data + i * (level + 1) * q->dim;
            float maxDist = 99999.9f;
            for (size_t j = 0; j < p_vecnum; ++j) {
                const float* vec_p = p->data + j * p->dim;
                float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
                dist_matrix[i][j] = dist;
                maxDist = std::min(maxDist, dist);
            }
            sum1 += maxDist;
        }


        //#pragma omp parallel for num_threads(4) reduction(+:sum2)
        #pragma omp simd reduction(+:sum2)
        for (size_t i = 0; i < p_vecnum; ++i) {
            float maxDist = 99999.9f;
            for (size_t j = 0; j < q_vecnum; ++j) {
                float dist = dist_matrix[j][i];
                maxDist = std::min(maxDist, dist);
            }
            sum2 += maxDist;
        }        
    }
    return sum1 / q_vecnum + sum2 / p_vecnum;
}

// static float L2SqrVecSet4Search(const vectorset* q, const vectorset* p, int level) {
//     float sum1 = 0.0f;
//     float sum2 = 0.0f;
//     // level = 0;
//     float (*L2Sqrfunc_)(const void*, const void*, const void*);
//     #if defined(USE_AVX512)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
//     #elif defined(USE_AVX)
//     L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
//     #else 
//     L2Sqrfunc_ = L2Sqr;
//     #endif

//     size_t p_vecnum = std::max(static_cast<size_t>(1), p->vecnum / (1 << level));

//     if (level != 0) {
//         std::vector<int> random_sequence_p;
//         for (int i = 0; i < p->vecnum; ++i) {
//             random_sequence_p.push_back(i); // 初始化为 1 到 n
//         }
//         // 使用随机数引擎打乱序列
//         std::random_device rd;  // 随机设备（种子）
//         std::mt19937 gen(rd()); // 使用 mt19937 引擎
//         std::shuffle(random_sequence_p.begin(), random_sequence_p.end(), gen);
//         std::vector<std::vector<float>> dist_matrix(q->vecnum, std::vector<float>(p_vecnum));
//         //#pragma omp parallel for num_threads(4) reduction(+:sum1)
//         #pragma omp simd reduction(+:sum1)
//         for (size_t i = 0; i < q->vecnum; ++i) {
//             const float* vec_q = q->data + i * q->dim;
//             float maxDist = 99999.9f;
//             for (size_t j = 0; j < p_vecnum; ++j) {
//                 const float* vec_p = p->data + random_sequence_p[j] * p->dim;
//                 float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
//                 dist_matrix[i][j] = dist;
//                 maxDist = std::min(maxDist, dist);
//             }
//             sum1 += maxDist;
//         }

//         //#pragma omp parallel for num_threads(4) reduction(+:sum2)
//         #pragma omp simd reduction(+:sum2)
//         for (size_t i = 0; i < p_vecnum; ++i) {
//             float maxDist = 99999.9f;
//             for (size_t j = 0; j < q->vecnum; ++j) {
//                 float dist = dist_matrix[j][i];
//                 maxDist = std::min(maxDist, dist);
//             }
//             sum2 += maxDist;
//         }
//     } else {
//         std::vector<std::vector<float>> dist_matrix(q->vecnum, std::vector<float>(p_vecnum));
//         //#pragma omp parallel for num_threads(4) reduction(+:sum1)
//         #pragma omp simd reduction(+:sum1)
//         for (size_t i = 0; i < q->vecnum; ++i) {
//             const float* vec_q = q->data + i * q->dim;
//             float maxDist = 99999.9f;
//             for (size_t j = 0; j < p_vecnum; ++j) {
//                 const float* vec_p = p->data + j * p->dim;
//                 float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
//                 dist_matrix[i][j] = dist;
//                 maxDist = std::min(maxDist, dist);
//             }
//             sum1 += maxDist;
//         }

//         //#pragma omp parallel for num_threads(4) reduction(+:sum2)
//         #pragma omp simd reduction(+:sum2)
//         for (size_t i = 0; i < p_vecnum; ++i) {
//             float maxDist = 99999.9f;
//             for (size_t j = 0; j < q->vecnum; ++j) {
//                 float dist = dist_matrix[j][i];
//                 maxDist = std::min(maxDist, dist);
//             }
//             sum2 += maxDist;
//         }
//     }

//     return sum1 / q->vecnum + sum2 /  p_vecnum;
// }


static float L2SqrVecSetInit(const vectorset* a, const vectorset* b, uint8_t* new_map, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    uint8_t fineEdgeTopk = 1;
    uint8_t fineEdgeMaxlen = 120;
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = L2Sqr;
    #endif
    uint8_t a_vecnum = (uint8_t) std::min(a->vecnum, (size_t)120);
    uint8_t b_vecnum = (uint8_t) std::min(b->vecnum, (size_t)120);

    std::vector<std::vector<float>> dist_matrix(a_vecnum, std::vector<float>(b_vecnum));
    //#pragma omp parallel for num_threads(4) reduction(+:sum1)
    // #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < a_vecnum; ++i) {
        const float* vec_a = a->data + i * a->dim;
        for (size_t j = 0; j < b_vecnum; ++j) {
            const float* vec_b = b->data + j * b->dim;
            float dist = L2Sqrfunc_(vec_a, vec_b, &b->dim);
            dist_matrix[i][j] = dist;
        }
    }
    // std::cout << (u_int16_t)a_vecnum << " " << (u_int16_t)b_vecnum << std::endl;
            
    std::priority_queue<std::pair<float, uint8_t>> pq;
    for (uint8_t i = 0; i < a_vecnum; ++i) {
        float maxDist = 99999.9f;
        for (uint8_t j = 0; j < b_vecnum; ++j) {
            if (pq.size() < fineEdgeTopk) {
                pq.push({dist_matrix[i][j], j});
            } else if (dist_matrix[i][j] < pq.top().first) {
                pq.pop();
                pq.push({dist_matrix[i][j], j});
            }
        }
        uint8_t j = pq.size();
        while (!pq.empty()) {
            j -= 1;
            if (pq.size() == 1) {
                maxDist = pq.top().first;
            }
            new_map[i * fineEdgeTopk + j] = pq.top().second;
            // std::cout << (u_int16_t)i << " " << (u_int16_t)j << " " << pq.top().first << " "  << (u_int16_t) pq.top().second << std::endl;
            pq.pop();
        }
        sum1 += maxDist;
    }

    for (uint8_t i = 0; i < b_vecnum; ++i) {
        float maxDist = 99999.9f;
        for (uint8_t j = 0; j < a_vecnum; ++j) {
            if (pq.size() < fineEdgeTopk) {
                pq.push({dist_matrix[j][i], j});
            } else if (dist_matrix[j][i] < pq.top().first) {
                pq.pop();
                pq.push({dist_matrix[j][i], j});
            }
        }
        uint8_t j = pq.size();
        while (!pq.empty()) {
            j -= 1;
            if (pq.size() == 1) {
                maxDist = pq.top().first;
            }
            new_map[i * fineEdgeTopk + fineEdgeMaxlen * fineEdgeTopk + j] = pq.top().second;
            // std::cout << (u_int16_t)i << " " << (u_int16_t)j << " " << pq.top().first << " "  << (u_int16_t) pq.top().second << std::endl;
            pq.pop();
        }
        sum2 += maxDist;
    }

    return sum1 / a_vecnum + sum2 / b_vecnum;
}

static float L2SqrVecSetMap(const vectorset* a, const vectorset* b, const vectorset* c, const uint8_t* old_map_ab, const uint8_t* old_map_bc, uint8_t* new_map, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = L2Sqr;
    #endif
    uint8_t a_vecnum = (uint8_t) std::min(a->vecnum, (size_t)120);
    uint8_t b_vecnum = (uint8_t) std::min(b->vecnum, (size_t)120);
    uint8_t c_vecnum = (uint8_t) std::min(c->vecnum, (size_t)120);
    std::vector<std::vector<float>> dist_matrix(a_vecnum, std::vector<float>(c_vecnum));

    uint8_t fineEdgeTopk = 1;
    uint8_t fineEdgeMaxlen = 120;
    // #pragma omp simd reduction(+:sum1)
    for (size_t i = 0; i < a_vecnum; ++i) {
        for (size_t j = 0; j < c_vecnum; ++j) {
            dist_matrix[i][j] = 9999.0;
        }
    }
    // std::cout << (u_int16_t)a_vecnum << " " << (u_int16_t)b_vecnum << " " << (u_int16_t)c_vecnum << std::endl;
    // #pragma omp simd reduction(+:sum1)
    for (uint8_t i = 0; i < a_vecnum; ++i) {
        for (uint8_t j = 0; j < fineEdgeTopk; j++) {
            for (uint8_t k = 0; k < fineEdgeTopk; k++) {
                uint8_t c_ind = old_map_bc[old_map_ab[i * fineEdgeTopk + j] * fineEdgeTopk + k];
                // std::cout << (u_int16_t)i << " " << (u_int16_t)old_map_ab[i * fineEdgeTopk + j] << " " << (u_int16_t)c_ind << std::endl;
                if (dist_matrix[i][c_ind] > 9000.0) {
                    const float* vec_a = a->data + i * a->dim;
                    const float* vec_c = c->data + c_ind * c->dim;
                    dist_matrix[i][c_ind] = L2Sqrfunc_(vec_a, vec_c, &a->dim);
                } 
            }
        }
    }
    // std::cout << "ac" << std::endl;

    // for (uint8_t i = 0; i < 10; ++i) {
    //     std::cout << (u_int16_t)i << " " << (u_int16_t)old_map_bc[i] << std::endl;
    // }
    // for (uint8_t i = 0; i < 10; ++i) {
    //     std::cout << (u_int16_t)i << " " << (u_int16_t)old_map_bc[i + 120 * 1] << std::endl;
    // }
    // std::cout << "== MapBC ==" << std::endl;

    // std::cout << (u_int16_t) a_vecnum << " " << (u_int16_t) c_vecnum << std::endl;
    // #pragma omp simd reduction(+:sum2)
    for (uint8_t i = 0; i < c_vecnum; ++i) {
        for (uint8_t j = 0; j < fineEdgeTopk; j++) {
            for (uint8_t k = 0; k < fineEdgeTopk; k++) {
                uint8_t a_ind = old_map_ab[old_map_bc[i * fineEdgeTopk + fineEdgeMaxlen * fineEdgeTopk + j] * fineEdgeTopk + fineEdgeMaxlen * fineEdgeTopk + k];
                // std::cout << (u_int16_t) i << " " << (u_int16_t) old_map_bc[i * fineEdgeTopk + fineEdgeMaxlen * fineEdgeTopk + j] << " " << (u_int16_t) a_ind << std::endl;
                if (dist_matrix[a_ind][i] > 9000.0) {
                    const float* vec_a = a->data + a_ind * a->dim;
                    const float* vec_c = c->data + i * c->dim;
                    dist_matrix[a_ind][i] = L2Sqrfunc_(vec_a, vec_c, &a->dim);
                } 
            }
        }
    }
    // std::cout << "ca" << std::endl;
    std::priority_queue<std::pair<float, uint8_t>> pq;
    for (uint8_t i = 0; i < a_vecnum; ++i) {
        float maxDist = 99999.9f;
        for (uint8_t j = 0; j < c_vecnum; ++j) {
            if (pq.size() < fineEdgeTopk) {
                pq.push({dist_matrix[i][j], j});
            } else if (dist_matrix[i][j] < pq.top().first) {
                pq.pop();
                pq.push({dist_matrix[i][j], j});
            }
        }
        uint8_t j = pq.size();
        while (!pq.empty()) {
            j -= 1;
            if (pq.size() == 1) {
                maxDist = pq.top().first;
            }
            new_map[i * fineEdgeTopk + j] = pq.top().second;
            pq.pop();
        }
        sum1 += maxDist;
    }

    for (uint8_t i = 0; i < c_vecnum; ++i) {
        float maxDist = 99999.9f;
        for (uint8_t j = 0; j < a_vecnum; ++j) {
            if (pq.size() < fineEdgeTopk) {
                pq.push({dist_matrix[j][i], j});
            } else if (dist_matrix[j][i] < pq.top().first) {
                pq.pop();
                pq.push({dist_matrix[j][i], j});
            }
        }
        uint8_t j = pq.size();
        while (!pq.empty()) {
            j -= 1;
            if (pq.size() == 1) {
                maxDist = pq.top().first;
            }
            new_map[i * fineEdgeTopk + fineEdgeMaxlen * fineEdgeTopk + j] = pq.top().second;
            // std::cout << "top1: " << (u_int16_t) pq.top().second << std::endl;
            pq.pop();
            
        }
        sum2 += maxDist;
    }
    // std::cout << pq.size() << std::endl;
    return sum1 / a_vecnum + sum2 / c_vecnum;
}

static float L2SqrVecSet4Search(const vectorset* q, const vectorset* p, int level) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    // level = 0;
    float (*L2Sqrfunc_)(const void*, const void*, const void*);
    #if defined(USE_AVX512)
    L2Sqrfunc_ = L2SqrSIMD16ExtAVX512;
    #elif defined(USE_AVX)
    L2Sqrfunc_ = L2SqrSIMD16ExtAVX;
    #else 
    L2Sqrfunc_ = L2Sqr;
    #endif

    size_t p_vecnum = std::max(static_cast<size_t>(1), p->vecnum / (1 << level));

    if (level != 0) {
        std::vector<int> random_sequence_p;
        for (int i = 0; i < p->vecnum; ++i) {
            random_sequence_p.push_back(i); // 初始化为 1 到 n
        }
        // 使用随机数引擎打乱序列
        std::random_device rd;  // 随机设备（种子）
        std::mt19937 gen(rd()); // 使用 mt19937 引擎
        std::shuffle(random_sequence_p.begin(), random_sequence_p.end(), gen);
        std::vector<std::vector<float>> dist_matrix(q->vecnum, std::vector<float>(p_vecnum));
        //#pragma omp parallel for num_threads(4) reduction(+:sum1)
        #pragma omp simd reduction(+:sum1)
        for (size_t i = 0; i < q->vecnum; ++i) {
            const float* vec_q = q->data + i * q->dim;
            float maxDist = 99999.9f;
            for (size_t j = 0; j < p_vecnum; ++j) {
                const float* vec_p = p->data + random_sequence_p[j] * p->dim;
                float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
                dist_matrix[i][j] = dist;
                maxDist = std::min(maxDist, dist);
            }
            sum1 += maxDist;
        }

        //#pragma omp parallel for num_threads(4) reduction(+:sum2)
        #pragma omp simd reduction(+:sum2)
        for (size_t i = 0; i < p_vecnum; ++i) {
            float maxDist = 99999.9f;
            for (size_t j = 0; j < q->vecnum; ++j) {
                float dist = dist_matrix[j][i];
                maxDist = std::min(maxDist, dist);
            }
            sum2 += maxDist;
        }
    } else {
        std::vector<std::vector<float>> dist_matrix(q->vecnum, std::vector<float>(p_vecnum));
        //#pragma omp parallel for num_threads(4) reduction(+:sum1)
        #pragma omp simd reduction(+:sum1)
        for (size_t i = 0; i < q->vecnum; ++i) {
            const float* vec_q = q->data + i * q->dim;
            float maxDist = 99999.9f;
            for (size_t j = 0; j < p_vecnum; ++j) {
                const float* vec_p = p->data + j * p->dim;
                float dist = L2Sqrfunc_(vec_q, vec_p, &p->dim);
                dist_matrix[i][j] = dist;
                maxDist = std::min(maxDist, dist);
            }
            sum1 += maxDist;
        }

        //#pragma omp parallel for num_threads(4) reduction(+:sum2)
        #pragma omp simd reduction(+:sum2)
        for (size_t i = 0; i < p_vecnum; ++i) {
            float maxDist = 99999.9f;
            for (size_t j = 0; j < q->vecnum; ++j) {
                float dist = dist_matrix[j][i];
                maxDist = std::min(maxDist, dist);
            }
            sum2 += maxDist;
        }
    }

    return sum1 / q->vecnum + sum2 /  p_vecnum;
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
