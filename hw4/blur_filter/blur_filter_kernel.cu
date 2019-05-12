/* Blur filter. Device code. */

#ifndef _BLUR_FILTER_KERNEL_H_
#define _BLUR_FILTER_KERNEL_H_

#include "blur_filter.h"

__global__ void 
blur_filter_kernel (const float *in, float *out, int size)
{
    
    int i,j;
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    int col = blockDim.x * blockX + threadX;
    int row = blockDim.y * blockY + threadY;
    int curr_row, curr_col;
    int num_neighbors = 0;
    float blur = 0.0;

    for (i = -1; i < 2; i++) {
        for (j = -1; j < 2; j++) {
            curr_row = row + i;
            curr_col = col + j;
            if ((curr_row > -1) && (curr_row < size) &&\
                (curr_col > -1) && (curr_col < size)) {
                blur += in[curr_row * size + curr_col];
                num_neighbors += 1;
            }
        }
    }

    out[row * size + col] = blur/num_neighbors;

    return;
}

#endif /* _BLUR_FILTER_KERNEL_H_ */
