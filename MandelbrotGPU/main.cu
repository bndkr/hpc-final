#include "bmp.h"
#include "computePixel.cuh"
#include "defs.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace
{
    __global__ void generateMandelbrot(unsigned char* pImage, const unsigned maxIterations)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < HEIGHT && j < WIDTH)
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
            pImageCopy[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = (unsigned char)sum_r;
            pImageCopy[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = (unsigned char)sum_g;
            pImageCopy[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = (unsigned char)sum_b;
        }
    }
} // namespace

int main(int, char**)
{
    unsigned char* hImageIn;
    unsigned char* hImageOut;
    cudaMalloc((void**)&hImageIn, HEIGHT * WIDTH * BYTES_PER_PIXEL);
    cudaMalloc((void**)&hImageOut, HEIGHT * WIDTH * BYTES_PER_PIXEL);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(ceil(WIDTH / (float)threadsPerBlock.x), ceil(HEIGHT / (float)threadsPerBlock.y));
    generateMandelbrot<<<numBlocks, threadsPerBlock>>>(hImageIn, 1000);
    cudaDeviceSynchronize();
    convolveMandelbrot<<<numBlocks, threadsPerBlock>>>(hImageIn, hImageOut);
    cudaDeviceSynchronize();

    unsigned char* pOutputImage = (unsigned char*)malloc(HEIGHT * WIDTH * BYTES_PER_PIXEL);
    cudaMemcpy(pOutputImage, hImageOut, HEIGHT * WIDTH * BYTES_PER_PIXEL, cudaMemcpyDeviceToHost);
    generateBitmapImage(pOutputImage, HEIGHT, WIDTH, "mandelbrot.bmp");
    printf("Image generated!!");
    free(pOutputImage);
    cudaFree(hImageIn);
    cudaFree(hImageOut);
    return 0;
}