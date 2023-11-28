#include "bmp.h"

#include <cstring>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define HEIGHT 1200
#define WIDTH 1200

#define BLACK 50
#define WHITE 230

namespace
{
    double calculatePixel(const double x_0, const double y_0, const unsigned maxIterations)
    {
        // cardioid check
        double p = sqrt((x_0 - 0.25) * (x_0 - 0.25) + y_0 * y_0);
        if (x_0 <= p - (2 * p * p) + 0.25)
            return -1;

        // period 2 bulb check
        if ((x_0 + 1) * (x_0 + 1) + y_0 * y_0 <= 1.0 / 16)
            return -1;

        double z_x = 0;
        double z_y = 0;

        double x_2 = 0;
        double y_2 = 0;

        unsigned int iteration = 0;

        while (x_2 + y_2 < 4 && iteration < maxIterations)
        {
            // iterate: z = z^2 + c
            z_y = 2 * z_x * z_y + y_0;
            z_x = x_2 - y_2 + x_0;
            x_2 = z_x * z_x;
            y_2 = z_y * z_y;
            iteration++;
        }
        if (iteration == maxIterations)
        {
            return -1; // inside the mandelbrot set
        }
        return iteration;
    }
} // namespace

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
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            int local_i = i - start_row;
            double iterations = calculatePixel(-2.0 + (j * 4.0 / WIDTH), (2.0 - (i * 4.0 / HEIGHT)), 100);
            if (iterations == -1)
            {
                pImageFragment[local_i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = BLACK; // red
                pImageFragment[local_i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = BLACK; // green
                pImageFragment[local_i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = BLACK; // blue
            }
            else
            {
                pImageFragment[local_i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = WHITE; // red
                pImageFragment[local_i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = WHITE; // green
                pImageFragment[local_i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = WHITE; // blue
            }
        }
    }
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
        for (int i = 1; i < rows_per_process + 1; i++)
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
