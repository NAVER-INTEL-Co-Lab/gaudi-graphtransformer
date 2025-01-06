#include "sparse_matrix_mul_fwd_f32_test.hpp"
#include "entry_points.hpp"
#include <cassert>



void SparseMatrixMulFwdF32Test::matrix_mul_reference_implementation(
        const float_3DTensor& input0,
        const float_3DTensor& input1,
        float_3DTensor& output)
{
    uint32_t batch_size = std::max(output.Size(2), 1u);

    for (int32_t batch = 0; batch < (int32_t)batch_size; batch++)
    {
        for (int32_t row = 0; row < (int32_t)output.Size(1); row++)
        {
            for (int32_t col = 0; col < (int32_t)output.Size(0); col++)
            {
                float accum = 0.0f;

                for (int32_t common = 0; common < (int32_t)input0.Size(0); common++)
                {
                    int32_t a_coord[] = {common, row, batch};
                    int32_t b_coord[] = {col, common, batch};

                    float a_val = input0.ElementAt(a_coord);
                    float b_val = input1.ElementAt(b_coord);

                    // Call templated mac to match the precision of TPC
                    //Ops::Mac<float, float>(&a_val, &b_val, &accum, &accum, NULL);
                    accum = std::fmaf(a_val, b_val, accum);
                }
                int32_t c_coord[] = {col, row, batch};

                output.SetElement(c_coord, accum);
            }
        }
    }
}

void SparseMatrixMulFwdF32Test::spmm_reference_implementation(
        const float_2DTensor& row_indices,
        const float_2DTensor& col_indices,
        const float_2DTensor& values,
        const float_3DTensor& bMatrix,
        float_3DTensor& output)
{
    uint32_t batch_size = std::max(output.Size(2), 1u);
    const int32_t num_nnz = row_indices.Size(0); // Assuming COO format

    // Assuming batch size = 1
    // Initialization of output C matrix 
    for (int32_t batch = 0; batch < (int32_t)batch_size; batch++){
        for(int32_t row = 0; row < (int32_t)output.Size(1); row++){
            for(int32_t col=0; col < (int32_t)output.Size(0); col++){
                int32_t c_coord[] = {col, row, batch};
                output.SetElement(c_coord, 0.0f);
            }
        }

        // sparse matrix multiplication

        for(int32_t nz_idx = 0 ; nz_idx < num_nnz; nz_idx++){
            int32_t row_coord[] = {nz_idx, 0};
            int32_t row = row_indices.ElementAt(row_coord);
            printf("row: %d\n", row);
            int32_t col_coord[] = {nz_idx,0};
            int common = static_cast<int>(col_indices.ElementAt(col_coord));
            int32_t val_coord[] = {nz_idx,0};
            float value = values.ElementAt(val_coord);

            for(int32_t dense_col = 0; dense_col < (int32_t)bMatrix.Size(0); dense_col++){
                int32_t b_coord[] = {dense_col, common, batch};
                float b_val = bMatrix.ElementAt(b_coord);
                int32_t c_coord[] = {dense_col, row, batch};
                float accum = output.ElementAt(c_coord);
                accum = std::fmaf(value, b_val, accum);
                output.SetElement(c_coord, accum);
                
            }
        }
    }
}

// int SparseMatrixMulFwdF32Test::runTest()
// {
//     const int col = 65;
//     const int row  = 6;
//     const int common = 4;
//     const int batch  = 1;

//     uint64_t fmInitializer_a[] = {common, row, batch};
//     uint64_t fmInitializer_b[] = {col, common, batch};
//     uint64_t fmInitializer_c[] = {col, row, batch};



//     float_3DTensor a_matrix(fmInitializer_a);
//     a_matrix.InitRand(1.0f, 10.0f);

//     float_3DTensor b_matrix(fmInitializer_b);
//     b_matrix.InitRand(1.0f, 10.0f);

//     float_3DTensor c_matrix(fmInitializer_c);
//     float_3DTensor c_matrix_ref(fmInitializer_c);


//     // SPMM initializaer

//     // nnz: 5, stored for 2D tensors for the batch size of 1
//     uint64_t sparseInitializer[] = {5, 1}; 
//     float_2DTensor row_indices(sparseInitializer);
//     float_2DTensor col_indices(sparseInitializer);
//     float_2DTensor values(sparseInitializer);

//     row_indices.SetElement({0, 0}, 0); col_indices.SetElement({0, 0}, 0); values.SetElement({0, 0}, 3.0f);
//     row_indices.SetElement({1, 0}, 1); col_indices.SetElement({1, 0}, 1); values.SetElement({1, 0}, 4.0f);
//     row_indices.SetElement({2, 0}, 2); col_indices.SetElement({2, 0}, 0); values.SetElement({2, 0}, 5.0f);
//     row_indices.SetElement({3, 0}, 2); col_indices.SetElement({3, 0}, 2); values.SetElement({3, 0}, 2.0f);
//     row_indices.SetElement({4, 0}, 3); col_indices.SetElement({4, 0}, 1); values.SetElement({4, 0}, 7.0f);

//     // execute reference implementation of the kernel.
//     spmm_mul_reference_implementation(a_matrix, b_matrix, c_matrix_ref);

    
//     std::cout << "Matrix multiply FWD F32 test pass!!" << std::endl;
//     return 0;
// }


// int SparseMatrixMulFwdF32Test::runTestSpmmResult()
// {
//     const int col = 65;
//     const int row = 6;
//     const int common = 4;
//     const int batch = 1;

//     uint64_t fmInitializer_a[] = {common, row, batch};
//     uint64_t fmInitializer_b[] = {col, common, batch};
//     uint64_t fmInitializer_c[] = {col, row, batch};

//     float_3DTensor a_matrix(fmInitializer_a);
//     a_matrix.InitRand(1.0f, 10.0f);

//     float_3DTensor b_matrix(fmInitializer_b);
//     b_matrix.InitRand(1.0f, 10.0f);

//     float_3DTensor c_matrix_dense(fmInitializer_c);
//     float_3DTensor c_matrix_sparse(fmInitializer_c);

//     // Create sparse representation of `a_matrix`
//     // nnz: 5, stored for 2D tensors for the batch size of 1
//     uint64_t sparseInitializer[] = {5, 1}; 
//     float_2DTensor row_indices(sparseInitializer);
//     float_2DTensor col_indices(sparseInitializer);
//     float_2DTensor values(sparseInitializer);

//     // Populate the sparse representation based on known sparsity
//     row_indices.SetElement({0, 0}, 0); col_indices.SetElement({0, 0}, 0); values.SetElement({0, 0}, a_matrix.ElementAt({0, 0, 0}));
//     row_indices.SetElement({1, 0}, 1); col_indices.SetElement({1, 0}, 1); values.SetElement({1, 0}, a_matrix.ElementAt({1, 1, 0}));
//     row_indices.SetElement({2, 0}, 2); col_indices.SetElement({2, 0}, 0); values.SetElement({2, 0}, a_matrix.ElementAt({0, 2, 0}));
//     row_indices.SetElement({3, 0}, 2); col_indices.SetElement({3, 0}, 2); values.SetElement({3, 0}, a_matrix.ElementAt({2, 2, 0}));
//     row_indices.SetElement({4, 0}, 3); col_indices.SetElement({4, 0}, 1); values.SetElement({4, 0}, a_matrix.ElementAt({1, 3, 0}));

//     // Run reference dense matrix multiplication
//     matrix_mul_reference_implementation(a_matrix, b_matrix, c_matrix_dense);

//     // Run sparse matrix multiplication
//     spmm_reference_implementation(row_indices, col_indices, values, b_matrix, c_matrix_sparse);

//     // Compare results
//     bool match = true;
//     for (int i = 0; i < c_matrix_dense.ElementCount(); i++) {
//         float dense_val = c_matrix_dense.Data()[i];
//         float sparse_val = c_matrix_sparse.Data()[i];
//         if (std::abs(dense_val - sparse_val) > 1e-6) {
//             std::cout << "Mismatch at element " << i << ": dense=" << dense_val << ", sparse=" << sparse_val << std::endl;
//             match = false;
//         }
//     }

//     if (match) {
//         std::cout << "SPMM matches dense matrix multiplication!" << std::endl;
//     } else {
//         std::cout << "SPMM does NOT match dense matrix multiplication!" << std::endl;
//         return -1;
//     }

//     std::cout << "Test passed!" << std::endl;
//     return 0;
// }


void testCompareDenseAndSparse() {
    const int col = 65;
    const int row = 6;
    const int common = 4;
    const int batch = 1;

    uint64_t fmInitializer_a[] = {common, row, batch};
    uint64_t fmInitializer_b[] = {col, common, batch};
    uint64_t fmInitializer_c[] = {col, row, batch};

    float_3DTensor a_matrix(fmInitializer_a);
    a_matrix.InitRand(1.0f, 10.0f);

    float_3DTensor b_matrix(fmInitializer_b);
    b_matrix.InitRand(1.0f, 10.0f);

    float_3DTensor c_matrix_dense(fmInitializer_c);
    float_3DTensor c_matrix_sparse(fmInitializer_c);

    // Sparse representation of `a_matrix`
    uint64_t sparseInitializer[] = {8, 1};
    float_2DTensor row_indices(sparseInitializer);
    float_2DTensor col_indices(sparseInitializer);
    float_2DTensor values(sparseInitializer);

    // Populate sparse representation based on known sparsity

    int row_indices_coords[] = {0, 0};
    // int col_indices_coords[] = {0, 0};
    // int values_coords[] = {0, 0};

    row_indices.SetElement(row_indices_coords, 0);
    // row_indices.SetElement({0, 0}, 0); col_indices.SetElement({0, 0}, 0); values.SetElement({0, 0}, a_matrix.ElementAt({0, 0, 0}));
    // row_indices.SetElement({1, 0}, 1); col_indices.SetElement({1, 0}, 1); values.SetElement({1, 0}, a_matrix.ElementAt({1, 1, 0}));
    // row_indices.SetElement({2, 0}, 2); col_indices.SetElement({2, 0}, 0); values.SetElement({2, 0}, a_matrix.ElementAt({0, 2, 0}));
    // row_indices.SetElement({3, 0}, 2); col_indices.SetElement({3, 0}, 2); values.SetElement({3, 0}, a_matrix.ElementAt({2, 2, 0}));
    // row_indices.SetElement({4, 0}, 3); col_indices.SetElement({4, 0}, 1); values.SetElement({4, 0}, a_matrix.ElementAt({1, 3, 0}));

    // Execute reference dense matrix multiplication
    SparseMatrixMulFwdF32Test testObj;
    testObj.matrix_mul_reference_implementation(a_matrix, b_matrix, c_matrix_dense);

    // Execute sparse matrix multiplication
    testObj.spmm_reference_implementation(row_indices, col_indices, values, b_matrix, c_matrix_sparse);

    // Compare outputs
    for (int i = 0; i < c_matrix_dense.ElementCount(); i++) {
        float dense_val = c_matrix_dense.Data()[i];
        float sparse_val = c_matrix_sparse.Data()[i];
        assert(std::abs(dense_val - sparse_val) < 1e-6 && "Mismatch in dense and sparse outputs");
    }

    std::cout << "Unit test passed: Dense and sparse implementations match!" << std::endl;
}


int main() {
    testCompareDenseAndSparse();
    return 0;
}
