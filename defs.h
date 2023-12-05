
// fun fact - 2520 is the smallest number divisible by all numbers 1-10
//            also it's the number with the most factors under 3000
#define HEIGHT 2520
#define WIDTH 2520

#define BLACK 50
#define WHITE 230

// edge detection kernel
#define KERNEL_CONTENT { { -1, -1, -1 }, { -1, 8, -1 }, { -1, -1, -1 } }

#ifdef __CUDACC__
__device__ char KERNEL[3][3] = KERNEL_CONTENT;
char HOST_KERNEL[3][3] = KERNEL_CONTENT;
#else
char KERNEL[3][3] = KERNEL_CONTENT;
#endif

#undef KERNEL_CONENT