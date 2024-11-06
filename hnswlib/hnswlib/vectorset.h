#pragma once
#include <cstdlib>
struct vectorset
{
    float* data = nullptr;
    int dim, vecnum;
    vectorset(float* _data, int _dim, int _vecnum):data(_data), dim(_dim), vecnum(_vecnum){}
    ~vectorset(){
        free(data);
    }
};
