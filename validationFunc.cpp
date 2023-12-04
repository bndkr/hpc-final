#include <cstdio>
#include <stdio.h>
#include <stdlib.h>

bool validateVSerial(const char* fileName, const char* fileName2, int Height, int Width);

int main(int argc, char** argv){
    if(argc != 5){
        printf("Five Args Required: validationFunc <firstFileName> <secondFileName> <Height> <Width>\n");
        return 1;
    }
    int height = atoi(argv[3]);
    int width = atoi(argv[4]);
    printf("%s and %s are equivalent: %d\n",argv[1], argv[2], validateVSerial(argv[1], argv[2], height, width));

    return 0;
}

bool validateVSerial(const char* fileName, const char* fileName2, int Height, int Width){
    //open file
    FILE *serialImg = fopen(fileName, "rb");

    // initialize serial image
    unsigned char * serialImgArray = (unsigned char *) malloc(Height * Width * 3);
    for(unsigned int i = 0; i < 1024*1024*3; i++){
        fread(&serialImgArray[i], 1, 1, serialImg);
    }
    fclose(serialImg);

    FILE *toValidateImg = fopen(fileName2, "rb");

    // initialize validate image
    unsigned char * toValidateImgArray = (unsigned char *) malloc(Height * Width * 3);
    for(unsigned int i = 0; i < 1024*1024*3; i++){
        fread(&toValidateImgArray[i], 1, 1, toValidateImg);
    }
    fclose(toValidateImg);

    for(int i = 0; i < Height * Width * 3; i++){
        if(serialImgArray[i] != toValidateImgArray[i]){
            return false;
        }
    }
    return true;
}