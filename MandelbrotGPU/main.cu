#include "bmp.h"
#include "computePixel.cuh"
#include "defs.h"
#include "timer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

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
    // Shared memory block for storing pixel values
    __device__ __shared__ unsigned char sharedBlock[(BLOCK_SIZE + 2) * (BLOCK_SIZE + 2)];

    // Calculate the indices of the current pixel
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy pixel values to shared memory (one channel at a time)
    for (int c = 0; c < BYTES_PER_PIXEL; c++)
    {
        sharedBlock[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 1] = pImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + c];

        // Copy border pixels
        if (threadIdx.x == 0 && j > 0)
            sharedBlock[(threadIdx.y + 1) * (blockDim.x + 2)] = pImage[i * WIDTH * BYTES_PER_PIXEL + (j - 1) * BYTES_PER_PIXEL + c];
        if (threadIdx.x == blockDim.x - 1 && j < WIDTH - 1)
            sharedBlock[(threadIdx.y + 1) * (blockDim.x + 2) + blockDim.x + 1] = pImage[i * WIDTH * BYTES_PER_PIXEL + (j + 1) * BYTES_PER_PIXEL + c];
        if (threadIdx.y == 0 && i > 0)
            sharedBlock[threadIdx.x + 1] = pImage[(i - 1) * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + c];
        if (threadIdx.y == blockDim.y - 1 && i < HEIGHT - 1)
            sharedBlock[(blockDim.y + 1) * (blockDim.x + 2) + threadIdx.x + 1] = pImage[(i + 1) * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + c];

        __syncthreads();

        // Convolution operation
        int sum = 0;
        for (int k = -1; k <= 1; k++)
        {
            for (int l = -1; l <= 1; l++)
            {
                sum += KERNEL[l + 1][k + 1] * sharedBlock[(threadIdx.y + k + 1) * (blockDim.x + 2) + threadIdx.x + l + 1];
            }
        }

        // Store the convolved pixel values in the output image
        pImageCopy[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + c] = (unsigned char)sum;

        __syncthreads();
    }
}
} // namespace

int main(int, char**)
{
    unsigned char* hImageIn;
    unsigned char* hImageOut;
    cudaMalloc((void**)&hImageIn, HEIGHT * WIDTH * BYTES_PER_PIXEL);
    cudaMalloc((void**)&hImageOut, HEIGHT * WIDTH * BYTES_PER_PIXEL);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(ceil(WIDTH / (float)threadsPerBlock.x), ceil(HEIGHT / (float)threadsPerBlock.y));
    generateMandelbrot<<<numBlocks, threadsPerBlock>>>(hImageIn, 1000);
    cudaDeviceSynchronize();
    timeval start = startTime();
    convolveMandelbrot<<<numBlocks, threadsPerBlock>>>(hImageIn, hImageOut);
    cudaDeviceSynchronize();

    unsigned char* pOutputImage = (unsigned char*)malloc(HEIGHT * WIDTH * BYTES_PER_PIXEL);
    cudaMemcpy(pOutputImage, hImageOut, HEIGHT * WIDTH * BYTES_PER_PIXEL, cudaMemcpyDeviceToHost);
    timeval end = stopTime();
    generateBitmapImage(pOutputImage, HEIGHT, WIDTH, "gpu.bmp");
    printf("%d,%f\n", BLOCK_SIZE, elapsedTime(start, end));
    free(pOutputImage);
    cudaFree(hImageIn);
    cudaFree(hImageOut);
    return 0;
}