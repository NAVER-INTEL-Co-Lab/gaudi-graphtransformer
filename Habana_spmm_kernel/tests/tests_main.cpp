/**********************************************************************
Copyright (c) 2024 Habana Labs.

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

#include <iostream>
#include "matrix_mul_fwd_f32_test.hpp"
#include "sparse_matrix_mul_fwd_f32_test.hpp"

int check_arg(int argc, char** argv, const char* device, const char* test)
{
    if( argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2], device) ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2], test) ==0))) ||
        (argc == 5 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2], device) ==0))  
        && (((strcmp(argv[3], "--test") ==0) || (strcmp(argv[3], "-t") ==0))
        && (strcmp(argv[4], test) ==0))) ||
        (argc == 5 && (((strcmp(argv[3], "--device") ==0) || (strcmp(argv[3], "-d") ==0))
        && (strcmp(argv[4], device) ==0))  
        && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2], test) ==0))))
        return 1;
    else
        return 0;
}
int main(int argc, char** argv)
{
    int result = 0;
    static int testCount = 0;

    if(argc == 2 && ((strcmp(argv[1], "--help") ==0) || (strcmp(argv[1],"-h") ==0)))
    {
        std::cout << argv[0] << " " << "[options]" << std::endl <<
            "Options:" << std::endl <<
            "N/A                        Run all test cases" << std::endl <<
            "-h | --help                Print this help" << std::endl <<
            "-d | --device <DeviceName> Run only kernels for the DeviceName" << std::endl <<
            "-t | --test  <TestName>    Run <TestName>> only   " << std::endl <<
            "DeviceName:" << std::endl <<
            "Gaudi                      Run all Gaudi kernels only   " << std::endl <<
            "Gaudi2                     Run all Gaudi2 kernels only   " << std::endl <<            
            "TestName:" << std::endl <<
            "FilterFwd2DBF16Test        Run FilterFwd2DBF16Test only   " << std::endl <<
            "SoftMaxBF16Test            Run SoftMaxBF16Test only   " << std::endl <<
            "CastGaudiTest              Run CastGaudiTest only   " << std::endl <<
            "BatchNormF32Test           Run BatchNormF32Test only   " << std::endl <<
            "LeakyReluF32GaudiTest      Run LeakyReluF32GaudiTest only   " << std::endl <<
            "SparseLengthsBF16Test      Run SparseLengthsBF16Test only   " << std::endl <<
            "CustomdivFwdF32Test        Run CustomdivFwdF32Test only   " << std::endl <<
            "Relu6FwdF32                Run Relu6FwdF32 only   " << std::endl <<
            "Relu6BwdF32                Run Relu6BwdF32 only   " << std::endl <<
            "Relu6FwdBF16               Run Relu6FwdBF16 only   " << std::endl <<
            "Relu6BwdBF16               Run Relu6BwdBF16 only   " << std::endl <<
            "ReluFwdF32                 Run ReluFwdF32 only   " << std::endl <<
            "ReluBwdF32                 Run ReluBwdF32 only   " << std::endl <<
            "ReluFwdBF16                Run ReluFwdBF16 only   " << std::endl <<
            "ReluBwdBF16                Run ReluBwdBF16 only   " << std::endl <<
            "MatrixMulFwdF32Test        Run MatrixMulFwdF32Test only   " << std::endl <<
            "SpatialConvF32Test         Run SpatialConvF32Test only   " << std::endl <<
            "SinF32Test                 Run SinF32Test only   " << std::endl <<
            "AddF32Test                 Run AddF32Test only   " << std::endl <<
            "AvgPool2DFwdF32Test        Run AvgPool2DFwdF32Test only   " << std::endl <<
            "AvgPool2DBwdF32Test        Run AvgPool2DBwdF32Test only   " << std::endl <<
            "SearchSortedFwdF32Test     Run SearchSortedFwdF32Test only   " << std::endl <<
            "GatherFwdDim0I32Test       Run GatherFwdDim0I32Test only   " << std::endl <<
            "KLDivFwdF32                Run KLDivFwdF32 only   "          << std::endl <<

            "AvgPool2DFwdF32Gaudi2Test  Run AvgPool2DFwdF32Gaudi2Test only   " << std::endl <<
            "AvgPool2DBwdF32Gaudi2Test  Run AvgPool2DBwdF32Gaudi2Test only   " << std::endl <<
            "CastF16toI16Gaudi2Test     Run CastF16toI16Gaudi2Test only   " << std::endl <<
            "SoftMaxBF16Gaudi2Test      Run SoftMaxBF16Gaudi2Test only   " << std::endl <<
            "UserLutGaudi2Test          Run UserLutGaudi2Test only   " << std::endl;

        exit(0);
    }
    else if(argc == 2) 
    {
        std::cout << "Please use --help or -h for more infomation" << std::endl;
        exit(0);
    }


    // if(check_arg(argc, argv, "Gaudi", "MatrixMulFwdF32Test"))
    // {
    //     MatrixMulFwdF32Test testMatrixMulFwdF32;
    //     testMatrixMulFwdF32.SetUp();
    //     result = testMatrixMulFwdF32.runTest();
    //     testMatrixMulFwdF32.TearDown();
    //     testCount ++;
    //     if (result != 0)
    //     {
    //         return result;
    //     }
    // }

    SparseMatrixMulFwdF32Test testSparseMatrixMulFwdF32;
    if(check_arg(argc, argv, "Gaudi2", "SparseMatrixMulFwdF32Test"))
    {
        testSparseMatrixMulFwdF32.SetUp();
        result = testSparseMatrixMulFwdF32.runTest();
        testSparseMatrixMulFwdF32.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }


    
    if(testCount > 0)
        std::cout << "All " << testCount  <<" tests pass!" <<std::endl;
    else
        std::cout << "Please use --help or -h for more infomation" << std::endl;
    return 0;
}
