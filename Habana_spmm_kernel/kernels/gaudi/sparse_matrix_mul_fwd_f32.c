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
#define PARTWIDTH  (ACCUMWIDTH * VECTORLENGTH)
#define PARTHEIGHT  6


void main(tensor row_indices,  // Row indices of sparse matrix A (CSR or COO format)
          tensor col_indices, // Column indices of sparse matrix A
          tensor values,      // Non-zero values of sparse matrix A
          tensor bMatrix,     // Dense matrix B
          tensor cMatrix,
          int ROW,
          int COL,
          int COMMON)     // Output dense matrix C
{
    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    const int numRows = get_dim_size(row_indices, 0); // Number of rows in sparse matrix A
    const int numColsB = get_dim_size(bMatrix, 1);   // Number of columns in dense matrix B
    const int numValues = get_dim_size(values, 0);   // Number of non-zero values in sparse matrix A

    int5 cCoords = {0};
    int5 aCoords = {0};
    int5 bCoords = {0};

    // Loop over rows of sparse matrix A
    for (int row = indexSpaceStart[0]; row < indexSpaceEnd[0]; row++) {
        int startIdx = s_i32_ld_tnsr(row_indices, row);     
        aCoords[0] += row;
        
        // I don't know how to do this but, read just one values at a time
        int aRow, aCol
        float aVal;

        aRow = s_i32_ld_tnsr(row_indices, row);
        aCol = s_i32_ld_tnsr(col_indices, row);
        aVal = s_f32_ld_tnsr(values,row);

            // Accumulate results for the row of output matrix C
            for (int j = indexSpaceStart[1]; j < indexSpaceEnd[1]; j+= VECTORLENGTH) {
                if ()
                bCoords[1] = j;
                float64 denseB = v_f32_ld_tnsr(bCoords, bMatrix); 

                accums[j] = v_f32_mac_b(denseB, value, accums[j]);
            }
        }

        // Store accumulated results in the output matrix C
        cCoords[0] = row;
        for (int j = 0; j < numColsB; j++) {
            cCoords[1] = j;
            v_f32_st_tnsr(cCoords, cMatrix, accums[j]);
        }


    }
}
