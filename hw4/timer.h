/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD
 * Timing operations.
 */

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

clock_t start, diff;

// Starts timer and resets the elapsed time
void timerStart(){
  start = clock();
}

// Stops the timer and returns elapsed time in msec
long timerStop(){
  diff = clock() - start;
  return (diff * 1000) / CLOCKS_PER_SEC;
}
