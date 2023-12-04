#include <cstring>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
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

void gpuGenerateMandelBrot(unsigned char* pImage, const unsigned maxIterations, int startRow, int endRow);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = HEIGHT / size;
    unsigned char* pMandelbrotImage;
    if (rank == 0)
    {
        pMandelbrotImage = (unsigned char*)malloc(HEIGHT * WIDTH * BYTES_PER_PIXEL);
        if (!pMandelbrotImage)
        {
            printf("Error allocating memory for image\n");

            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    unsigned char* pImageFragment = (unsigned char*)malloc(rows_per_process * WIDTH * BYTES_PER_PIXEL);
    int start_row = rank * rows_per_process;
    int end_row = start_row + rows_per_process;
    printf("rank %d: start_row %d, end_row %d\n", rank, start_row, end_row);
    fflush(stdout);
    // for (int i = start_row; i < end_row; i++)
    // {
    //     for (int j = 0; j < WIDTH; j++)
    //     {
    //         int local_i = i - start_row;
    //         double iterations = calculatePixel(-2.0 + (j * 4.0 / WIDTH), (2.0 - (i * 4.0 / HEIGHT)), 100);
    //         if (iterations == -1)
    //         {
    //             pImageFragment[local_i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = BLACK; // red
    //             pImageFragment[local_i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = BLACK; // green
    //             pImageFragment[local_i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = BLACK; // blue
    //         }
    //         else
    //         {
    //             pImageFragment[local_i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = WHITE; // red
    //             pImageFragment[local_i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = WHITE; // green
    //             pImageFragment[local_i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = WHITE; // blue
    //         }
    //     }
    // }
    gpuGenerateMandelBrot(pImageFragment, 1000, start_row, end_row);
    printf("rank %d: done generating mandelbrot pixels.\n", rank);
    fflush(stdout);
    MPI_Gather(pImageFragment, rows_per_process * WIDTH * BYTES_PER_PIXEL, MPI_UNSIGNED_CHAR, rank == 0 ? pMandelbrotImage : nullptr, rows_per_process * WIDTH * BYTES_PER_PIXEL, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // +2 for the extra top and bottom rows needed for convolution
    unsigned char* pConvoFragment = (unsigned char*)malloc((rows_per_process + 2) * WIDTH * BYTES_PER_PIXEL);
    unsigned char* pConvoFragmentResult = (unsigned char*)malloc(rows_per_process * WIDTH * BYTES_PER_PIXEL);

    // use MPI to send the data to each process
    if (rank == 0)
    {
        // start at one so we get a row of blank pixels for convolution
        for (int i = 1; i < rows_per_process + 2; i++)
        {
            for (int j = 0; j < WIDTH; j++)
            {
                pConvoFragment[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = pMandelbrotImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2];
                pConvoFragment[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = pMandelbrotImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1];
                pConvoFragment[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = pMandelbrotImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0];
            }
        }
        for (int i = 1; i < size; i++)
        {
            // ternary operator so we dont send a non-existent row to the last process
            printf("rank %d: sending %d rows to process %d\n", rank, rows_per_process + (i == size - 1 ? 1 : 2), i);
            MPI_Send(pMandelbrotImage + (i * rows_per_process - 1) * WIDTH * BYTES_PER_PIXEL, (rows_per_process + (i == size - 1 ? 1 : 2)) * WIDTH * BYTES_PER_PIXEL, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        printf("rank %d: receiving %d rows\n", rank, rows_per_process + (rank == size - 1 ? 1 : 2));
        MPI_Recv(pConvoFragment, (rows_per_process + (rank == size - 1 ? 1 : 2)) * WIDTH * BYTES_PER_PIXEL, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    char kernel[3][3] = { { -1, -1, -1 }, { -1, 8, -1 }, { -1, -1, -1 } };
    // iterate over the fragment and apply the kernel
    for (int i = 0; i < rows_per_process; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            int sum_r = 0;
            int sum_g = 0;
            int sum_b = 0;
            int count = 0;
            for (int k = -1; k <= 1; k++)
            {
                for (int l = -1; l <= 1; l++)
                {
                    int local_i = i + 1; // +1 to offset the top row
                    if (local_i + k >= 0 && local_i + k < rows_per_process + 2 && j + l >= 0 && j + l < WIDTH)
                    {
                        sum_r += kernel[l + 1][k + 1] * pConvoFragment[(local_i + k) * WIDTH * BYTES_PER_PIXEL + (j + l) * BYTES_PER_PIXEL + 2];
                        sum_g += kernel[l + 1][k + 1] * pConvoFragment[(local_i + k) * WIDTH * BYTES_PER_PIXEL + (j + l) * BYTES_PER_PIXEL + 1];
                        sum_b += kernel[l + 1][k + 1] * pConvoFragment[(local_i + k) * WIDTH * BYTES_PER_PIXEL + (j + l) * BYTES_PER_PIXEL];
                        count += 1;
                    }
                }
            }
            pConvoFragmentResult[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = (unsigned char)sum_r;
            pConvoFragmentResult[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = (unsigned char)sum_g;
            pConvoFragmentResult[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = (unsigned char)sum_b;
        }
    }
    unsigned char* pFinalImage;
    if (rank == 0)
    {
        pFinalImage = (unsigned char*)malloc(HEIGHT * WIDTH * BYTES_PER_PIXEL);
    }
    MPI_Gather(pConvoFragmentResult, rows_per_process * WIDTH * BYTES_PER_PIXEL, MPI_UNSIGNED_CHAR, rank == 0 ? pFinalImage : nullptr, rows_per_process * WIDTH * BYTES_PER_PIXEL, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        generateBitmapImage(pMandelbrotImage, HEIGHT, WIDTH, "mandelbrot.bmp");
        generateBitmapImage(pFinalImage, HEIGHT, WIDTH, "convolved.bmp");
        printf("rank %d: done generating images\n", rank);
        fflush(stdout);
        free(pMandelbrotImage);
        free(pFinalImage);
    }
    free(pConvoFragment);
    free(pConvoFragmentResult);
    printf("exiting rank %d\n", rank);
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
