#include <math.h>

__device__ inline double calculatePixel(const double x_0, const double y_0, const unsigned maxIterations)
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