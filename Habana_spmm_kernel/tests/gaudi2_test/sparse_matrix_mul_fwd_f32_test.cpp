#include "sparse_matrix_mul_fwd_f32_test.hpp"
#include "entry_points.hpp"
#include <cassert>

void SparseMatrixMulFwdF32Test::matrix_mul_reference_implementation(
        const float_2DTensor& input0,
        const float_2DTensor& input1,
        float_2DTensor& output)
{

    for (int32_t row = 0; row < (int32_t)output.Size(1); row++)
    {
        for (int32_t col = 0; col < (int32_t)output.Size(0); col++)
        {
            float accum = 0.0f;

            for (int32_t common = 0; common < (int32_t)input0.Size(0); common++)
            {
                int32_t a_coord[] = {common, row};
                int32_t b_coord[] = {col, common};

                float a_val = input0.ElementAt(a_coord);
                float b_val = input1.ElementAt(b_coord);

                // Call templated mac to match the precision of TPC
                //Ops::Mac<float, float>(&a_val, &b_val, &accum, &accum, NULL);
                accum = std::fmaf(a_val, b_val, accum);
            }
            int32_t c_coord[] = {col, row};

            output.SetElement(c_coord, accum);
        }
    }
}

void SparseMatrixMulFwdF32Test::spmm_reference_implementation(
        const int32_2DTensor& row_indices,
        const int32_2DTensor& col_indices,
        const float_2DTensor& values,
        const float_2DTensor& bMatrix,
        float_2DTensor& output)
{
    const int32_t num_nnz = row_indices.Size(0); // Assuming COO format

    // Assuming batch size = 1
    // Initialization of output C matrix 
    for(int32_t row = 0; row < (int32_t)output.Size(1); row++){
        for(int32_t col=0; col < (int32_t)output.Size(0); col++){
            int32_t c_coord[] = {col, row};
            output.SetElement(c_coord, 0.0f);
        }
    }

    // sparse matrix multiplication

    for(int32_t nz_idx = 0 ; nz_idx < num_nnz; nz_idx++){
        int32_t row_coord[] = {nz_idx, 0};
        int32_t row = row_indices.ElementAt(row_coord);
        // int32_t col = col_indices.ElementAt(row_coord);
        // float_t val = values.ElementAt(row_coord);

        // printf("row: %d\n", row);
        // printf("col: %d\n", col);
        // printf("val: %f\n", val);

        int32_t col_coord[] = {nz_idx,0};
        int common = static_cast<int>(col_indices.ElementAt(col_coord));
        int32_t val_coord[] = {nz_idx,0};
        float value = values.ElementAt(val_coord);

        for(int32_t dense_col = 0; dense_col < (int32_t)bMatrix.Size(0); dense_col++){
            int32_t b_coord[] = {dense_col, common};
            float b_val = bMatrix.ElementAt(b_coord);
            // printf("B_matrix col: %f\n", b_val);
            int32_t c_coord[] = {dense_col, row};
            float accum = output.ElementAt(c_coord);
            accum = std::fmaf(value, b_val, accum);
            output.SetElement(c_coord, accum);
            
        }
    }
}

int SparseMatrixMulFwdF32Test::runTest(){
    const int col = 65;
    const int row = 6;
    const int common = 4;

    uint64_t fmInitializer_a[] = {common, row};
    uint64_t fmInitializer_b[] = {col, common};
    uint64_t fmInitializer_c[] = {col, row};

    float_2DTensor a_matrix(fmInitializer_a);
    a_matrix.InitRand(1.0f, 10.0f);

    float_2DTensor b_matrix(fmInitializer_b);
    b_matrix.InitRand(10.0f, 30.0f);

    float_2DTensor c_matrix(fmInitializer_c);
    float_2DTensor c_matrix_dense(fmInitializer_c);
    float_2DTensor c_matrix_ref(fmInitializer_c);

    uint64_t nnz = 0;
    for(int i=0;i<row;i++){
        for(int j=0;j<common;j++){
            int32_t coord[] = {j, i};
            if(a_matrix.ElementAt(coord) != 0){
                nnz++;
            }
        }
    }

    // Sparse representation of `a_matrix`
    uint64_t sparseInitializer[] = {nnz, 1};
    int32_2DTensor row_indices(sparseInitializer);
    int32_2DTensor col_indices(sparseInitializer);
    float_2DTensor values(sparseInitializer);

    // Convert dense matrix A into COO format

    int cur_indices_coords[] = {0, 0};
    for(int i=0;i<row;i++){
        for(int j=0;j<common;j++){
            int32_t coord[] = {j, i};
            if(a_matrix.ElementAt(coord) > 1e-6){
                row_indices.SetElement(cur_indices_coords, i);
                col_indices.SetElement(cur_indices_coords, j);
                values.SetElement(cur_indices_coords, a_matrix.ElementAt(coord));
                cur_indices_coords[0]++;
            }
        }
    }

    // Execute sparse matrix multiplication
    matrix_mul_reference_implementation(a_matrix, b_matrix, c_matrix_dense);
    spmm_reference_implementation(row_indices, col_indices, values, b_matrix, c_matrix_ref);
    // std::cout << "Dense Matrix" << std::endl;
    // c_matrix_dense.Print(0);
    // std::cout << "Dense Matrix" << std::endl;

    // c_matrix_ref.Print(0);
    // for (int element = 0 ; element <  c_matrix_ref.ElementCount() ; element++)
    // {
    //     if (abs(c_matrix_dense.Data()[element] - c_matrix_ref.Data()[element]) > 1e-6)
    //     {
    //         std::cout << "Wrong Elemet at: " << element << " Expected: " << c_matrix_ref.Data()[element] << " Got: " << c_matrix_dense.Data()[element] << std::endl;
    //         std::cout << "Sparse Matrix multiply FWD F32 test failed!!" << std::endl;
    //         return -1;
    //     }
    // }

    // generate input for query call
    m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;

    m_in_defs.inputTensorNr = 4;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), row_indices);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), col_indices);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), values);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[3]), b_matrix);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), c_matrix);

    tpc_lib_api::GuidInfo *guids = nullptr;
    unsigned kernelCount = 0;
    tpc_lib_api::GlueCodeReturn result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);
    guids = new tpc_lib_api::GuidInfo[kernelCount];
    result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);

    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.guid.name, guids[GAUDI2_KERNEL_SPARSE_MATRIXMUL_FWD_F32].name);
    result  = InstantiateTpcKernel(&m_in_defs,&m_out_defs);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }
    //Generate and load tnesor descriptors

    std::vector<TensorDesc2> vec;
    vec.push_back(row_indices.GetTensorDescriptor());
    vec.push_back(col_indices.GetTensorDescriptor());
    vec.push_back(values.GetTensorDescriptor());
    vec.push_back(b_matrix.GetTensorDescriptor());
    vec.push_back(c_matrix.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(guids, kernelCount);

    c_matrix.Print(0);

    c_matrix_ref.Print(0);
    std::setprecision(6);
    for (int element = 0 ; element <  c_matrix_ref.ElementCount() ; element++)
    {
        if (abs(c_matrix.Data()[element] - c_matrix_ref.Data()[element]) > 1e-6)
        {
            std::cout << "Wrong Elemet at: " << element << " Expected: " << c_matrix_ref.Data()[element] << " Got: " << c_matrix.Data()[element] << std::endl;
            std::cout << "Sparse Matrix multiply FWD F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Sparse Matrix multiply FWD F32 test pass!!" << std::endl;
    return 0;
}

// To Check the Graunuality of the dataset
// int SparseMatrixMulFwdF32Test::runTest(){
//     const int col = 3;
//     const int row = 5;
//     const int common = 4;
//     const int batch = 1;

//     uint64_t fmInitializer_a[] = {common, row};
//     uint64_t fmInitializer_b[] = {col, common};
//     uint64_t fmInitializer_c[] = {col, row};

//     float_2DTensor a_matrix(fmInitializer_a);
//     a_matrix.InitRand(1.0f, 10.0f);

//     float_2DTensor b_matrix(fmInitializer_b);
//     b_matrix.InitRand(1.0f, 10.0f);

//     float_2DTensor c_matrix(fmInitializer_c);
//     float_2DTensor c_matrix_ref(fmInitializer_c);

//     // Make a matrix sparse 
//     int sparse_coord[] = {0,0};
//     a_matrix.SetElement(sparse_coord,0);
//     sparse_coord[1] = 0 ; sparse_coord[0] = 2; a_matrix.SetElement(sparse_coord,0);
//     sparse_coord[1] = 1 ; sparse_coord[0] = 0; a_matrix.SetElement(sparse_coord,0);
//     sparse_coord[1] = 1 ; sparse_coord[0] = 1; a_matrix.SetElement(sparse_coord,0);
//     sparse_coord[1] = 1 ; sparse_coord[0] = 3; a_matrix.SetElement(sparse_coord,0);
//     sparse_coord[1] = 2 ; sparse_coord[0] = 0; a_matrix.SetElement(sparse_coord,0);
//     sparse_coord[1] = 2 ; sparse_coord[0] = 1; a_matrix.SetElement(sparse_coord,0);
//     sparse_coord[1] = 2 ; sparse_coord[0] = 2; a_matrix.SetElement(sparse_coord,0);
//     sparse_coord[1] = 2 ; sparse_coord[0] = 3; a_matrix.SetElement(sparse_coord,0);
//     sparse_coord[1] = 3 ; sparse_coord[0] = 1; a_matrix.SetElement(sparse_coord,0);
//     sparse_coord[1] = 3 ; sparse_coord[0] = 2; a_matrix.SetElement(sparse_coord,0);
//     sparse_coord[1] = 3 ; sparse_coord[0] = 3; a_matrix.SetElement(sparse_coord,0);

//     // Sparse representation of `a_matrix`
//     uint64_t sparseInitializer[] = {8, 1};
//     int32_2DTensor row_indices(sparseInitializer);
//     int32_2DTensor col_indices(sparseInitializer);
//     float_2DTensor values(sparseInitializer);

//     // Convert dense matrix A into COO format

//     int cur_indices_coords[] = {0, 0};
//     for(int i=0;i<row;i++){
//         for(int j=0;j<common;j++){
//             for(int k=0;k<batch;k++){
//                 int32_t coord[] = {j, i, k};
//                 if(a_matrix.ElementAt(coord) != 0){
//                     row_indices.SetElement(cur_indices_coords, i);
//                     col_indices.SetElement(cur_indices_coords, j);
//                     values.SetElement(cur_indices_coords, a_matrix.ElementAt(coord));
//                     cur_indices_coords[0]++;
//                 }
//             }
//         }
//     }

//     // Execute sparse matrix multiplication
//     spmm_reference_implementation(row_indices, col_indices, values, b_matrix, c_matrix_ref);

//     // generate input for query call
//     m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;

//     m_in_defs.inputTensorNr = 4;
//     LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), row_indices);
//     LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), col_indices);
//     LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), values);
//     LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[3]), b_matrix);

//     m_in_defs.outputTensorNr = 1;
//     LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), c_matrix);

//     tpc_lib_api::GuidInfo *guids = nullptr;
//     unsigned kernelCount = 0;
//     tpc_lib_api::GlueCodeReturn result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);
//     guids = new tpc_lib_api::GuidInfo[kernelCount];
//     result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);

//     if (result != tpc_lib_api::GLUE_SUCCESS)
//     {
//         std::cout << "Can't get kernel name!! " << result << std::endl;
//         ReleaseKernelNames(guids, kernelCount);
//         return -1;
//     }

//     strcpy(m_in_defs.guid.name, guids[GAUDI2_KERNEL_SPARSE_MATRIXMUL_FWD_F32].name);
//     result  = InstantiateTpcKernel(&m_in_defs,&m_out_defs);
//     if (result != tpc_lib_api::GLUE_SUCCESS)
//     {
//         std::cout << "Glue test failed, can't load kernel " << result << std::endl;
//         ReleaseKernelNames(guids, kernelCount);
//         return -1;
//     }
//     //Generate and load tnesor descriptors

//     std::vector<TensorDesc2> vec;
//     vec.push_back(row_indices.GetTensorDescriptor());
//     vec.push_back(col_indices.GetTensorDescriptor());
//     vec.push_back(values.GetTensorDescriptor());
//     vec.push_back(b_matrix.GetTensorDescriptor());
//     vec.push_back(c_matrix.GetTensorDescriptor());

//     // execute a simulation of the kernel using TPC simulator,
//     TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
//     ReleaseKernelNames(guids, kernelCount);

//     c_matrix.Print(0);

//     c_matrix_ref.Print(0);
//     for (int element = 0 ; element <  c_matrix_ref.ElementCount() ; element++)
//     {
//         if (abs(c_matrix.Data()[element] - c_matrix_ref.Data()[element]) > 5*1e-5)
//         {
//             std::cout << "Wrong Elemet at: " << element << " Expected: " << c_matrix_ref.Data()[element] << " Got: " << c_matrix.Data()[element] << std::endl;
//             std::cout << "Sparse Matrix multiply FWD F32 test failed!!" << std::endl;
//             return -1;
//         }
//     }
//     std::cout << "Sparse Matrix multiply FWD F32 test pass!!" << std::endl;
//     return 0;
// }


void testCompareDenseAndSparse(){
    const int col = 3;
    const int row = 5;
    const int common = 4;
    const int batch = 1;

    uint64_t fmInitializer_a[] = {common, row};
    uint64_t fmInitializer_b[] = {col, common};
    uint64_t fmInitializer_c[] = {col, row};

    float_2DTensor a_matrix(fmInitializer_a);
    a_matrix.InitRand(1.0f, 10.0f);

    float_2DTensor b_matrix(fmInitializer_b);
    b_matrix.InitRand(1.0f, 10.0f);

    float_2DTensor c_matrix_dense(fmInitializer_c);
    float_2DTensor c_matrix_sparse(fmInitializer_c);


    // Make a matrix sparse 
    int sparse_coord[] = {0,0};
    a_matrix.SetElement(sparse_coord,0);
    sparse_coord[1] = 0 ; sparse_coord[0] = 2; a_matrix.SetElement(sparse_coord,0);
    sparse_coord[1] = 1 ; sparse_coord[0] = 0; a_matrix.SetElement(sparse_coord,0);
    sparse_coord[1] = 1 ; sparse_coord[0] = 1; a_matrix.SetElement(sparse_coord,0);
    sparse_coord[1] = 1 ; sparse_coord[0] = 3; a_matrix.SetElement(sparse_coord,0);
    sparse_coord[1] = 2 ; sparse_coord[0] = 0; a_matrix.SetElement(sparse_coord,0);
    sparse_coord[1] = 2 ; sparse_coord[0] = 1; a_matrix.SetElement(sparse_coord,0);
    sparse_coord[1] = 2 ; sparse_coord[0] = 2; a_matrix.SetElement(sparse_coord,0);
    sparse_coord[1] = 2 ; sparse_coord[0] = 3; a_matrix.SetElement(sparse_coord,0);
    sparse_coord[1] = 3 ; sparse_coord[0] = 1; a_matrix.SetElement(sparse_coord,0);
    sparse_coord[1] = 3 ; sparse_coord[0] = 2; a_matrix.SetElement(sparse_coord,0);
    sparse_coord[1] = 3 ; sparse_coord[0] = 3; a_matrix.SetElement(sparse_coord,0);

    for(int i=0;i<row;i++){
        for(int j=0;j<common;j++){
            for(int k=0;k<1;k++){
                int32_t coord[] = {j, i, k};
                printf("a_matrix[%d][%d][%d]: %f\n", i, j, k, a_matrix.ElementAt(coord));
            }
        }
    }

    // Sparse representation of `a_matrix`
    uint64_t sparseInitializer[] = {8, 1};
    int32_2DTensor row_indices(sparseInitializer);
    int32_2DTensor col_indices(sparseInitializer);
    float_2DTensor values(sparseInitializer);

    // Convert dense matrix A into COO format

    int cur_indices_coords[] = {0, 0};
    for(int i=0;i<row;i++){
        for(int j=0;j<common;j++){
            for(int k=0;k<batch;k++){
                int32_t coord[] = {j, i, k};
                if(a_matrix.ElementAt(coord) != 0){
                    row_indices.SetElement(cur_indices_coords, i);
                    col_indices.SetElement(cur_indices_coords, j);
                    values.SetElement(cur_indices_coords, a_matrix.ElementAt(coord));
                    cur_indices_coords[0]++;
                }
            }
        }
    }

    // DEBUG
    // for(int i=0;i<8;i++){
    //     int32_t coord[] = {i, 0};
    //     printf("row_indices[%d][%d]: %f\n", i, 0, row_indices.ElementAt(coord));
    // }
    
    // for(int i=0;i<8;i++){
    //     int32_t coord[] = {i, 0};
    //     printf("col_indices[%d][%d]: %f\n", i, 0, col_indices.ElementAt(coord));
    // }

    // for(int i=0;i<8;i++){
    //     int32_t coord[] = {i, 0};
    //     printf("values[%d][%d]: %f\n", i, 0, values.ElementAt(coord));
    // }
    
    // Execute reference dense matrix multiplication
    SparseMatrixMulFwdF32Test testObj;
    testObj.matrix_mul_reference_implementation(a_matrix, b_matrix, c_matrix_dense);

    // Execute sparse matrix multiplication
    testObj.spmm_reference_implementation(row_indices, col_indices, values, b_matrix, c_matrix_sparse);


    //Compare with values
    for (int i = 0; i < c_matrix_dense.ElementCount(); i++) {
        float dense_val = c_matrix_dense.Data()[i];
        float sparse_val = c_matrix_sparse.Data()[i];
        if (abs(dense_val - sparse_val) > 1e-6){
            std::cout << "Sparse Matrix multiply FWD F32 test failed!!" << std::endl;
        }

    }

    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            for(int k=0;k<batch;k++){
                int32_t coord[] = {j, i, k};
                printf("c_matrix_dense[%d][%d][%d]: %f\n", i, j, k, c_matrix_dense.ElementAt(coord));
                printf("c_matrix_sparse[%d][%d][%d]: %f\n", i, j, k, c_matrix_sparse.ElementAt(coord));
            }
        }
    }

    c_matrix_dense.Print(0);
    c_matrix_sparse.Print(0);
    c_matrix_dense.Print(1);
    c_matrix_sparse.Print(1);
    c_matrix_dense.Print(2);
    c_matrix_sparse.Print(2);
    c_matrix_dense.Print(3);
    c_matrix_sparse.Print(3);
    c_matrix_dense.Print(4);
    c_matrix_sparse.Print(4);
    
    std::cout << "Unit test passed: Dense and sparse implementations match!" << std::endl;

}


// int main() {
//     testCompareDenseAndSparse();
//     return 0;
// }
