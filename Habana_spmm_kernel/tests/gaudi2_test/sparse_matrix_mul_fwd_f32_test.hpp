/**********************************************************************
Copyright (c) 2021 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef SPARSE_MATRIX_MUL_FWD_F32_TEST_HPP
#define SPARSE_MATRIX_MUL_FWD_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "sparse_matrix_mul_fwd_f32.hpp"


class SparseMatrixMulFwdF32Test : public TestBase
{
public:
    SparseMatrixMulFwdF32Test() {}
    ~SparseMatrixMulFwdF32Test() {}
    int runTest();

    inline static void matrix_mul_reference_implementation(
            const float_2DTensor& input0,
            const float_2DTensor& input1,
            float_2DTensor& output);
    
    inline static void spmm_reference_implementation(
            const int32_2DTensor& row_indices,
            const int32_2DTensor& col_indices,
            const float_2DTensor& values,
            const float_2DTensor& b_matrix,
            float_2DTensor& output);
private:
    SparseMatrixMulFwdF32Test(const SparseMatrixMulFwdF32Test& other) = delete;
    SparseMatrixMulFwdF32Test& operator=(const SparseMatrixMulFwdF32Test& other) = delete;


};

#endif /* SPARSE_MATRIX_MUL_FWD_F32_TEST_HPP */

