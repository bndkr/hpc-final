#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C"
{
#endif
struct timeval startTime();

struct timeval stopTime();

float elapsedTime(struct timeval startTime, struct timeval endTime);
#ifdef __cplusplus
}
#endif

#endif