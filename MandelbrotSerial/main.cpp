#include "bmp.h"
#include "computePixel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define HEIGHT 2000
#define WIDTH 2000

#define CONVOLVE

#define BLACK 50
#define WHITE 230

int main(int, char**)
{
    unsigned char* pImage = (unsigned char*)malloc(HEIGHT * WIDTH * BYTES_PER_PIXEL);
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

    // blur kernel
    // unsigned char kernel[3][3] = { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } };
    // sharpen kernel
    // char kernel[3][3] = { { 0, -1, 0 }, { -1, 5, -1 }, { 0, -1, 0 } };
    // edge detection kernel
    char kernel[3][3] = { { -1, -1, -1 }, { -1, 8, -1 }, { -1, -1, -1 } };

    unsigned char* pImageCopy = (unsigned char*)malloc(HEIGHT * WIDTH * BYTES_PER_PIXEL);

    // for each pixel
    for (i = 0; i < HEIGHT; i++)
    {
        for (j = 0; j < WIDTH; j++)
        {
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
                        count += 1;
                    }
                }
            }
#ifdef CONVOLVE
            pImageCopy[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = (unsigned char)sum_r;
            pImageCopy[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = (unsigned char)sum_g;
            pImageCopy[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = (unsigned char)sum_b;
#endif
        }
    }
    generateBitmapImage(pImageCopy, HEIGHT, WIDTH, "mandelbrot.bmp");
    printf("Image generated!!");
    return 0;
}