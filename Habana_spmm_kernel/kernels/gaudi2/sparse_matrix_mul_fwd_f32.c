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
#pragma tpc_printf (enable)
#define PRINTF_ENABLE 1

#define VECTORLENGTH 64
#define ACCUMWIDTH  1

void main(tensor row_indices,  // Row indices of sparse matrix A (CSR or COO format)
          tensor col_indices, // Column indices of sparse matrix A
          tensor values,      // Non-zero values of sparse matrix A
          tensor bMatrix,     // Dense matrix B
          tensor cMatrix) 
{
    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    // const int numRows = get_dim_size(row_indices, 0); // Number of rows in sparse matrix A
    // const int numColsB = get_dim_size(bMatrix, 1);   // Number of columns in dense matrix B
    // const int numValues = get_dim_size(values, 0);   // Number of non-zero values in sparse matrix A

    int5 cCoords = {0,0,0,0,0};
    int5 aCoords = {0,0,0,0,0};
    int5 bCoords = {0,0,0,0,0};

    printf("\n\n\n\n");
    printf("Current Index Start\n");
    printf("%d ", indexSpaceStart[0]);
    printf("%d ", indexSpaceStart[1]);
    printf("%d ", indexSpaceStart[2]);
    printf("%d \n", indexSpaceStart[3]);


    printf("Current Index End\n");
    printf("%d ", indexSpaceEnd[0]);
    printf("%d ", indexSpaceEnd[1]);
    printf("%d ", indexSpaceEnd[2]);
    printf("%d \n", indexSpaceEnd[3]);


    // Loop over row vectors sparse matrix A
    for (int sparseIdx = indexSpaceStart[3]; sparseIdx < indexSpaceEnd[3]; sparseIdx++) {
        aCoords[0] = sparseIdx;
        // printf("Sparse Idx: %d\n", sparseIdx);
        
        // printf("Acoords\n");
        // printf("%d ", aCoords[0]);
        // printf("%d ", aCoords[1]);
        // printf("%d ", aCoords[2]);
        // printf("%d \n", aCoords[3]);
        // I don't know how to do this but, read just one values at a time
        // int aRow = v_i32_ld_tnsr_partial_b(aCoords, row_indices);
        // __global__ int* validCountAddr = (__global__ int*)gen_addr(validCountCoord, validCount);
        __global__ int * p_aValue0 = gen_addr(aCoords, row_indices);
        __global__ int * p_aValue1 = gen_addr(aCoords, col_indices);
        __global__ float * p_aValue2 = gen_addr(aCoords, values);

        int aRow = s_i32_ld_g(p_aValue0);
        int aCol = s_i32_ld_g(p_aValue1);
        float64 aVal = s_f32_ld_g(p_aValue2);

        // printf("Loaded values: ");
        // printf("aRow: %d\n", aRow);
        // printf("aCol: %d\n", aCol);
        // printf("aVal: %f\n", aVal[0]);

        // Accumulate results for the row of output matrix C
        for (int denseCol = indexSpaceStart[0]; denseCol < indexSpaceEnd[0]; denseCol+= VECTORLENGTH) {
        // Need to fix it 
            bCoords[0] = denseCol;
            cCoords[0]  = denseCol;

            for(int bRow = indexSpaceStart[2]; bRow < indexSpaceEnd[2]; bRow++) {
                if (aCol != bRow) {
                    continue;
                }

                bCoords[1] = bRow;

                for(int cRow = indexSpaceStart[1]; cRow < indexSpaceEnd[1]; cRow++){
                    if (aRow != cRow) {
                        continue;
                    }

                    cCoords[1] = cRow;

                    float64 denseB = v_f32_ld_tnsr_b(bCoords, bMatrix); 
                    float64 accum = v_f32_ld_tnsr_b(cCoords, cMatrix);
                    float64 result = v_f32_mul_b(denseB, aVal);
                    result += accum;
                    // v_f32_st_tnsr(cCoords, cMatrix, result);
                    v_f32_st_tnsr_partial(cCoords, cMatrix, result, 64-1, 0);
                }
            }
        }
    }
}

