#include <math.h>
#include "bmp.h"
#include "computePixel.cuh"
#include "defs.h"

__global__ void generateMandelbrot(unsigned char* pImage, const unsigned maxIterations, int startRow, int endRow)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HEIGHT && j < WIDTH && i >= startRow && j <= endRow)
    {
        double iterations = calculatePixel(-2.0 + (j * 4.0 / WIDTH), (2.0 - (i * 4.0 / HEIGHT)), maxIterations);
        if (iterations == -1)
        {
            pImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = BLACK; // red
            pImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = BLACK; // green
            pImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = BLACK; // blue
        }
        else
        {
            pImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = WHITE; // red
            pImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = WHITE; // green
            pImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = WHITE; // blue
        }
    }
}

__global__ void convolveMandelbrot(unsigned char* pImage, unsigned char* pImageCopy)
{
    // edge detection kernel
    char kernel[3][3] = { { -1, -1, -1 }, { -1, 8, -1 }, { -1, -1, -1 } };
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HEIGHT && j < WIDTH)
    {
        int sum_r = 0;
        int sum_g = 0;
        int sum_b = 0;
        for (int k = -1; k <= 1; k++)
        {
            for (int l = -1; l <= 1; l++)
            {
                if (i + k >= 0 && i + k < HEIGHT && j + l >= 0 && j + l < WIDTH)
                {
                    sum_r += kernel[l + 1][k + 1] * pImage[(i + k) * WIDTH * BYTES_PER_PIXEL + (j + l) * BYTES_PER_PIXEL + 2];
                    sum_g += kernel[l + 1][k + 1] * pImage[(i + k) * WIDTH * BYTES_PER_PIXEL + (j + l) * BYTES_PER_PIXEL + 1];
                    sum_b += kernel[l + 1][k + 1] * pImage[(i + k) * WIDTH * BYTES_PER_PIXEL + (j + l) * BYTES_PER_PIXEL];
                }
            }
        }
        pImageCopy[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = (unsigned char) sum_r;
        pImageCopy[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = (unsigned char) sum_g;
        pImageCopy[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = (unsigned char) sum_b;
    }
}

void gpuGenerateMandelBrot(unsigned char* pImage, const unsigned maxIterations, int startRow, int endRow) {
    int height = endRow - startRow;
    cudaMalloc((void**) &pImage, (height) * WIDTH * BYTES_PER_PIXEL);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(ceil(WIDTH / (float)threadsPerBlock.x), ceil(height / (float)threadsPerBlock.y));
    generateMandelbrot<<<numBlocks, threadsPerBlock>>>(pImage, 1000, startRow, endRow);
    cudaDeviceSynchronize();
}

void gpuConvolveImage(unsigned char* pImage, unsigned char* pImageCopy) {
}