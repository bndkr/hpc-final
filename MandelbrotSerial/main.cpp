#include "bmp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

#define HEIGHT 200
#define WIDTH 200

#define BLACK 0
#define WHITE 255

int main(int, char**)
{
    unsigned char* pImage = (unsigned char*)malloc(HEIGHT * WIDTH * BYTES_PER_PIXEL);
    char* imageFileName = (char*)"bitmapImage.bmp";

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

    // perform a blur image convolution kernel

    // define a 3x3 kernel
    // blur kernel
    // unsigned char kernel[3][3] = { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } };

    // sharpen kernel
    unsigned char kernel[3][3] = { { 0, -1, 0 }, { -1, 5, -1 }, { 0, -1, 0 } };

    // for each pixel
    for (i = 0; i < HEIGHT; i++)
    {
        for (j = 0; j < WIDTH; j++)
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
            pImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 2] = sum_r / count;
            pImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 1] = sum_g / count;
            pImage[i * WIDTH * BYTES_PER_PIXEL + j * BYTES_PER_PIXEL + 0] = sum_b / count;
        }
    }

    generateBitmapImage(pImage, HEIGHT, WIDTH, imageFileName);
    printf("Image generated!!");
    return 0;
}