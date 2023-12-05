#include "timer.h"

struct timeval startTime() {
    struct timeval startTime;
    gettimeofday(&startTime, NULL);
    return startTime;
}

struct timeval stopTime() {
    struct timeval endTime;
    gettimeofday(&endTime, NULL);
    return endTime;
}

float elapsedTime(struct timeval startTime, struct timeval endTime) {
    return ((float) ((endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec)/1.0e6));
}