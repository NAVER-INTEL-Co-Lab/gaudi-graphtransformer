/**********************************************************************
Copyright (c) 2020 Habana Labs.

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

#include <vector>
#include <cstring>
#include <iostream>
#include "sparse_matrix_mul_fwd_f32.hpp"


extern unsigned char _binary___sparse_matrix_mul_fwd_f32_o_start;
extern unsigned char _binary___sparse_matrix_mul_fwd_f32_o_end;
tpc_lib_api::GlueCodeReturn SparseMatrixMulFwdF32::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_sparse_matrix_multiply_fwd_f32");
     return tpc_lib_api::GLUE_SUCCESS;
 }


tpc_lib_api::GlueCodeReturn SparseMatrixMulFwdF32::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;
    // const int c_unrollCount = 4;

    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 4)
    {
        in_defs->inputTensorNr  = 4;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr !=1)
    {
        in_defs->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    //validate matrix dimensions
    //Row Col Values for 2D float Tensors, B_Matrix 3D tensor
    // if ((in_defs->inputTensors[0].geometry.dims != 2 &&
    //      in_defs->inputTensors[0].geometry.dims != 3) ||
    //     (in_defs->inputTensors[1].geometry.dims != 2 &&
    //      in_defs->inputTensors[1].geometry.dims != 3) ||
    //     in_defs->inputTensors[0].geometry.maxSizes[0] != in_defs->inputTensors[1].geometry.maxSizes[1])
    // {
    //     return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    // }
    // if ((in_defs->outputTensors[0].geometry.dims != 2 &&
    //      in_defs->outputTensors[0].geometry.dims != 3) ||
    //     in_defs->outputTensors[0].geometry.maxSizes[0] != in_defs->inputTensors[1].geometry.maxSizes[0] ||
    //     in_defs->outputTensors[0].geometry.maxSizes[1] != in_defs->inputTensors[0].geometry.maxSizes[1])
    // {
    //     return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    // }

    // validate input and output data type
    // Check 4 inputs and output data types
    if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_I32 ||
        in_defs->inputTensors[1].geometry.dataType != tpc_lib_api::DATA_I32 ||
        in_defs->inputTensors[2].geometry.dataType != tpc_lib_api::DATA_F32 ||
        in_defs->inputTensors[3].geometry.dataType != tpc_lib_api::DATA_F32 ||
        in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32
        )
    {
        in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_I32;
        in_defs->inputTensors[1].geometry.dataType = tpc_lib_api::DATA_I32;
        in_defs->inputTensors[2].geometry.dataType = tpc_lib_api::DATA_F32;
        in_defs->inputTensors[3].geometry.dataType = tpc_lib_api::DATA_F32;
        in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;

        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II-IV -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    int32_t c_vlen             = 64;

    // If we split same tensor size as GEMM multiplications we can use the same index space geometry
    // MODIFICATION HERE SPLIT WITH INDEX
    out_defs->indexSpaceRank = 4;

    // [Ccol == Bcol, Crow, Brow, Acol]
    out_defs->indexSpaceGeometry[0] =
        (in_defs->outputTensors[0].geometry.maxSizes[0] + c_vlen - 1) / c_vlen;
    out_defs->indexSpaceGeometry[1] = (in_defs->outputTensors[0].geometry.maxSizes[1]);
    out_defs->indexSpaceGeometry[2] = (in_defs->inputTensors[3].geometry.maxSizes[1]);
    out_defs->indexSpaceGeometry[3] = (in_defs->inputTensors[0].geometry.maxSizes[0]);

    // Matrix C - Tensor Access Pattern
    out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim     = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].a = c_vlen;
    out_defs->outputTensorAccessPattern[0].mapping[0].start_b = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].end_b   = c_vlen - 1;

    out_defs->outputTensorAccessPattern[0].mapping[1].indexSpaceDim     = 1;
    out_defs->outputTensorAccessPattern[0].mapping[1].a = 1;
    out_defs->outputTensorAccessPattern[0].mapping[1].start_b = 0;
    out_defs->outputTensorAccessPattern[0].mapping[1].end_b   = 1 - 1;

    // Sparse splits [8X1] 

    for (int i=0; i<3;i++){
        out_defs->inputTensorAccessPattern[i].mapping[0].indexSpaceDim  = 3;
        out_defs->inputTensorAccessPattern[i].mapping[0].a = 1;
        out_defs->inputTensorAccessPattern[i].mapping[0].start_b = 0;
        out_defs->inputTensorAccessPattern[i].mapping[0].end_b = 0;

        // We use only one column
        // If we fused row col value vectors we can mapping indexSpace DIm
        out_defs->inputTensorAccessPattern[i].mapping[1].indexSpaceDim     = 1;
        out_defs->inputTensorAccessPattern[i].mapping[1].a = 0;
        out_defs->inputTensorAccessPattern[i].mapping[1].start_b = 0;
        out_defs->inputTensorAccessPattern[i].mapping[1].end_b   = 0;
    }


    // Matrix B - Tensor Access Pattern - 4th inputTensors
    out_defs->inputTensorAccessPattern[3].mapping[0].indexSpaceDim     = 0;
    out_defs->inputTensorAccessPattern[3].mapping[0].a = c_vlen;
    out_defs->inputTensorAccessPattern[3].mapping[0].start_b = 0;
    out_defs->inputTensorAccessPattern[3].mapping[0].end_b   = c_vlen - 1;

    out_defs->inputTensorAccessPattern[3].mapping[1].indexSpaceDim     = 2;
    out_defs->inputTensorAccessPattern[3].mapping[1].a = 1;
    out_defs->inputTensorAccessPattern[3].mapping[1].start_b = 0;
    out_defs->inputTensorAccessPattern[3].mapping[1].end_b = 1 - 1;

    // HELP HERE - How to split the tensor access pattern for the sparse matrix

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___sparse_matrix_mul_fwd_f32_o_end - &_binary___sparse_matrix_mul_fwd_f32_o_start);
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernel.kernelElf,
                &_binary___sparse_matrix_mul_fwd_f32_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

 
