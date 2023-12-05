#include "bmp.h"
#include "computePixel.h"
#include "defs.h"
#include "timer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

void convolveImage(unsigned char* pImage, unsigned char* rImage, char kernel[3][3], size_t my_start_y, size_t my_start_x, size_t my_end_y, size_t my_end_x)
{
    if (my_end_y > HEIGHT)
    {
        my_end_y = HEIGHT;
    }
    if (my_end_x > WIDTH)
    {
        my_end_x = WIDTH;
    }
    for (auto i = my_start_y; i < my_end_y; i++)
    {
        for (auto j = my_start_x; j < my_end_x; j++)
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
            rImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = sum_r;
            rImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = sum_g;
            rImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = sum_b;
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Usage: ./MandelbrotThreaded <number of threads>\n");
        return 1;
    }

    unsigned char* pImage = (unsigned char*)malloc(HEIGHT * WIDTH * BYTES_PER_PIXEL);
    unsigned char* rImage = (unsigned char*)malloc(HEIGHT * WIDTH * BYTES_PER_PIXEL);
    char* imageFileName = "threaded.bmp";
    int numOfThreads = atoi(argv[1]);
    int tileWidth = WIDTH;
    int tileHeight = ceil(HEIGHT / numOfThreads);
    int xtiles = 1;
    int ytiles = ceil(HEIGHT / tileHeight);
    int totalTiles = xtiles * ytiles;

    int i, j;
    for (i = 0; i < HEIGHT; i++)
    {
        for (j = 0; j < WIDTH; j++)
        {
            double iterations = calculatePixel(-2.0 + (j * 4.0 / WIDTH), (2.0 - (i * 4.0 / HEIGHT)), 100);
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

    printf("Done Calculating Image\n");

    std::vector<std::thread> threads;
    timeval start = startTime();
    printf("number of tiles: %d\n", totalTiles);
    for (size_t i = 0; i < ytiles; i++)
    {
        for (size_t j = 0; j < xtiles; j++)
        {
            threads.push_back(std::thread(convolveImage, pImage, rImage, KERNEL, i * tileHeight, j * tileWidth, i * tileHeight + tileHeight, j * tileWidth + tileWidth));
        }
    }
    while (!threads.empty())
    {
        threads.back().join();
        threads.pop_back();
    }
    timeval end = stopTime();

    generateBitmapImage(rImage, HEIGHT, WIDTH, imageFileName);
    printf("Image generated!! In %f seconds\n", elapsedTime(start, end));
    return 0;
}