/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD
 * This is a skeleton for implementing prefix sum using GPU, inspired
 * by nvidia course of similar name.
 */

#include <stdio.h>
#include "timer.h"
#include <math.h>
#include <string.h>

#define N 512

/*
 * You should implement the simple scan function here!
 */
__global__ void scan_simple(float *g_odata, float *g_idata, int n) {
  extern  __shared__  float temp[];

  // STUDENT: YOUR CODE GOES HERE.
  g_odata[threadIdx.x] = 0.0;

}

/*
 * You should implement the prescan kernel function here!
 */
__global__ void prescan(float *g_odata, float *g_idata, int n) {
  extern  __shared__  float temp[];  

  // STUDENT: YOUR CODE GOES HERE.

}

/*
 * Fills an array a with n random floats.
 */
void random_floats(float* a, int n) {
  float d;
  // Comment out this line if you want consistent "random".
  srand(time(NULL));
  for (int i = 0; i < n; ++i) {
    d = rand() % 8;
    a[i] = ((rand() % 64) / (d > 0 ? d : 1));
  }
}

/*
 * Simple Serial implementation of scan.
 */
void serial_scan(float* out, float* in, int n) {
  float total_sum = 0;
  out[0] = 0;
  for (int i = 1; i < n; i++) {
    total_sum += in[i-1];
    out[i] = out[i-1] + in[i-1];
  }
  if (total_sum != out[n-1]) {
    printf("Warning: exceeding accuracy of float.\n");
  }
}

/*
 * This is a simple function that confirms that the output of the scan
 * function matches that of a golden image (array).
 */
bool printError(float *gold_out, float *test_out, bool show_all) {
  bool firstFail = true;
  bool error = false;
  float epislon = 0.1;
  float diff = 0.0;
  for (int i = 0; i < N; ++i) {
    diff = abs(gold_out[i] - test_out[i]);
    if ((diff > epislon) && firstFail) {
      printf("ERROR: gold_out[%d] = %f != test_out[%d] = %f // diff = %f \n", i, gold_out[i], i, test_out[i], diff);
      firstFail = show_all;
      error = true;
    }
  }
  return error;
}

int main(void) {
  float *in, *out, *gold_out; // host
  float *d_in, *d_out; // device
  int size = sizeof(float) * N;

  timerStart();
  cudaMalloc((void **)&d_in, size);
  cudaMalloc((void **)&d_out, size);
  
  in = (float *)malloc(size);
  random_floats(in, N);
  out = (float *)malloc(size);
  gold_out = (float *)malloc(size);
  printf("TIME: Init took %d ms\n",  timerStop());
  // ***********
  // RUN SERIAL SCAN
  // ***********
  timerStart();
  serial_scan(gold_out, in, N);
  printf("TIME: Serial took %d ms\n",  timerStop());

  timerStart();
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
  printf("TIME: Copy took %d ms\n",  timerStop());
  // ***********
  // RUN SIMPLE SCAN
  // ***********
  timerStart();
  scan_simple<<< 1, 512, N * 2 * sizeof(float)>>>(d_out, d_in, N);
  cudaDeviceSynchronize();
  printf("TIME: Simple kernel took %d ms\n",  timerStop());
  timerStart();
  cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
  printf("TIME: Copy back %d ms\n",  timerStop());

  if (printError(gold_out, out, false)) {
    printf("ERROR: The simple scan function failed to produce proper output.\n");
  } else {
    printf("CONGRATS: The simple scan function produced proper output.\n");
  }

  // ***********
  // RUN PRESCAN
  // note size change in number of threads, only need 256 because each
  // thread should handle 2 elements.
  // ***********
  timerStart();
  prescan<<< 1, 256, N * 2 * sizeof(float)>>>(d_out, d_in, N);
  cudaDeviceSynchronize();
  printf("TIME: Prescan kernel took %d ms\n",  timerStop());
  timerStart();
  cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
  printf("TIME: Copy back %d ms\n",  timerStop());

  if (printError(gold_out, out, false)) {
    printf("ERROR: The prescan function failed to produce proper output.\n");
  } else {
    printf("CONGRATS: The prescan function produced proper output.\n");
  }

  return 0;
}
