#ifndef BMP_H
#define BMP_H

const int BYTES_PER_PIXEL = 3; // red, green, & blue
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

#ifdef __cplusplus
extern "C"
{
#endif

    void generateBitmapImage(unsigned char* image, int height, int width, const char* imageFileName);
    unsigned char* createBitmapFileHeader(int height, int stride);
    unsigned char* createBitmapInfoHeader(int height, int width);

#ifdef __cplusplus
}
#endif

#endif