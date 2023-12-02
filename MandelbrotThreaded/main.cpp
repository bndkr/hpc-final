#include "bmp.h"
#include "computePixel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

#define HEIGHT 1083
#define WIDTH 1083

#define CONVOLVE

#define BLACK 0
#define WHITE 255

void convolveImage(unsigned char* pImage, unsigned char* rImage, unsigned char kernel[3][3], size_t my_start_y, size_t my_start_x, size_t my_end_y, size_t my_end_x)
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
            // for each pixel, average the 3x3 grid around it
            int sum_r = 0;
            int sum_g = 0;
            int sum_b = 0;
            int count = 0;
            for (int k = -1; k <= 1; k++)
            {
                for (int l = -1; l <= 1; l++)
                {
                    if (i + k >= 0 && i + k < HEIGHT && j + l >= 0 && j + l < WIDTH)
                    {
                        sum_r += kernel[l + 1][k + 1] * pImage[(i + k) * WIDTH * BYTES_PER_PIXEL + (j + l) * BYTES_PER_PIXEL + 2];
                        sum_g += kernel[l + 1][k + 1] * pImage[(i + k) * WIDTH * BYTES_PER_PIXEL + (j + l) * BYTES_PER_PIXEL + 1];
                        sum_b += kernel[l + 1][k + 1] * pImage[(i + k) * WIDTH * BYTES_PER_PIXEL + (j + l) * BYTES_PER_PIXEL];
                        count += kernel[l + 1][k + 1];
                    }
                }
            }
            rImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = sum_r / count;
            rImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = sum_g / count;
            rImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = sum_b / count;
        }
    }
}

int main(int, char**)
{
    unsigned char* pImage = (unsigned char*)malloc(HEIGHT * WIDTH * BYTES_PER_PIXEL);
    unsigned char* rImage = (unsigned char*)malloc(HEIGHT * WIDTH * BYTES_PER_PIXEL);
    char* imageFileName = (char*)"bitmapImage.bmp";
    float numOfThreads = 5;
    int xtiles = (int)ceil((float)WIDTH / numOfThreads);
    int ytiles = (int)ceil((float)HEIGHT / numOfThreads);
    int totalTiles = xtiles * ytiles;
    int tileWidth = (int)(ceil((float)WIDTH / xtiles));
    int tileHeight = (int)(ceil((float)HEIGHT / ytiles));

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

    // perform a blur image convolution kernel

    // define a 3x3 kernel
    // blur kernel
    unsigned char kernel[3][3] = { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } };

    // sharpen kernel
    // unsigned char kernel[3][3] = { { 0, 1, 0 }, { 1, -4, 1 }, { 0, 1, 0 } };

    // for each pixel
    // std::thread my_thread(convolveImage, pImage, rImage, kernel, 0, 0, 400, 400);
    // std::thread my_thread2(convolveImage, pImage, rImage, kernel, 0, 400, 400, 800);
    // std::thread my_thread3(convolveImage, pImage, rImage, kernel, 400, 0, 800, 400);
    // std::thread my_thread4(convolveImage, pImage, rImage, kernel, 400, 400, 800, 800);
    // my_thread.join();
    // my_thread2.join();
    // my_thread3.join();
    // my_thread4.join();

    std::vector<std::thread> threads;
    for (size_t i = 0; i < ytiles; i++)
    {
        for (size_t j = 0; j < xtiles; j++)
        {
            threads.push_back(std::thread(convolveImage, pImage, rImage, kernel, i * tileHeight, j * tileWidth, i * tileHeight + tileHeight, j * tileWidth + tileWidth));
        }
        while (!threads.empty())
        {
            threads.back().join();
            threads.pop_back();
        }
    }

    generateBitmapImage(rImage, HEIGHT, WIDTH, imageFileName);
    printf("Image generated!!");
    return 0;
}