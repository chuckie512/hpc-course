/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD
 * This is a skeleton for implementing prefix sum using GPU, inspired
 * by nvidia course of similar name.
 * 
 * student: Charles Smith <cas275@pitt.edu>
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
 
  int pin=1, pout=0; //keep track of where in the temp array we are.  0 for first half, 1 for second
   
  if(threadIdx.x<511)
    temp[threadIdx.x+1] = g_idata[threadIdx.x];


  int offset; //how far away we're adding from
  for(offset=1; offset<n; offset*=2){

    //swap the buffers
    pin  =1 - pin ;
    pout =1 - pout;

    //in general we try to avoid branching on GPUs as it is a giant decrease to preformance
    //but there's not to many ways around this one
    if(threadIdx.x>=offset)//we have work to do!
      temp[pout*n+threadIdx.x] = temp[pin*n+threadIdx.x] + temp[pin*n+threadIdx.x - offset];  //sum
    else //we already found the answer, let's keep track of it
      temp[pout*n+threadIdx.x] = temp[pin*n+threadIdx.x];

    __syncthreads(); //don't want to do work before everyone is ready

  }
  //great! we have the answer! (i think...) let's copy it to the output buffer

  g_odata[threadIdx.x] = temp[pout*n+threadIdx.x];

}

/*
 * You should implement the prescan kernel function here!
 */
__global__ void prescan(float *g_odata, float *g_idata, int n) {
  extern  __shared__  float temp[];  

//to many headaches working on this.... I looked at the linked code and started fresh.

//  g_odata[threadIdx.x] = g_idata[threadIdx.x];
//  
//  // STUDENT: YOUR CODE GOES HERE.
//    temp[threadIdx.x] = g_idata[threadIdx.x];
//
//  int offset = 1; //how far to move the data
//  //upsweep
//  __syncthreads();//get everyone together
//  for(; offset<n; offset*=2){
//    if(!(threadIdx.x+1)%(offset*2))
//      temp[threadIdx.x] += temp[threadIdx.x-offset];
//    __syncthreads();
//    printf("offset = %d, temp = %f, threadIdx.x = %d\n",offset, temp[threadIdx.x], threadIdx.x);
//  }
//  
//  if(threadIdx.x==0)
//    temp[n-1] = 0;  //clear the last entry
//
//  offset/=2;
//  __syncthreads();
//
//  //downsweep
//  for(; offset>=1;offset/=2){
//    if(!(threadIdx.x+1)%(offset*2)){
//      //swaps
//      float t = temp[threadIdx.x-offset];
//      temp[threadIdx.x-offset] = temp[threadIdx.x];
//      temp[threadIdx.x]+=t;
//    
//    }
//    __syncthreads();
//    printf("offset = %d, temp = %f, threadIdx.x = %d\n",offset, temp[threadIdx.x], threadIdx.x);
//    __syncthreads();
//  }
//  printf("temp[%d] = %f\n", threadIdx.x, temp[threadIdx.x]); 
//  g_odata[threadIdx.x]=temp[threadIdx.x];
//  printf("g_odata[%d] = %f\n", threadIdx.x, g_odata[threadIdx.x]);


  int offset = 1;
  temp[2*threadIdx.x] = g_idata[2*threadIdx.x]; 
  temp[2*threadIdx.x+1] = g_idata[2*threadIdx.x+1];
  for(int i = n/2; i > 0; i/=2){ 
  
    __syncthreads();
    if(threadIdx.x < i) {
      int ai = offset*(2*threadIdx.x+1)-1;
      int bi = offset*(2*threadIdx.x+2)-1;
      temp[bi] += temp[ai]; 
    }
    offset *= 2;
  }
  if(threadIdx.x == 0) { 
    temp[n - 1] = 0; 
  } 
  for(int i = 1; i < n; i *= 2) {
    offset /= 2;
    __syncthreads();
    if(threadIdx.x < i){
      int ai = offset*(2*threadIdx.x+1)-1;
      int bi = offset*(2*threadIdx.x+2)-1;
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
    __syncthreads();
  }
  __syncthreads();
  //HEY LOOK IT COPIES THIS TIME AHHHHHHHHH
  g_odata[2*threadIdx.x] = temp[2*threadIdx.x]; 
  g_odata[2*threadIdx.x+1] = temp[2*threadIdx.x+1]; 


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

  if (printError(gold_out, out, true)) {
    printf("ERROR: The simple scan function failed to produce proper output.\n");
    //printf("produced output:\n");
    //for(int i=0; i<512; i++){
    //  printf("%d: %f\n",i, out[i]);
    //}
  } else {
    printf("CONGRATS: The simple scan function produced proper output.\n");
  }
  
  out[0]=-2;
  out[1]=-2;
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

  if (printError(gold_out, out, true)) {
    printf("ERROR: The prescan function failed to produce proper output.\n");
  } else {
    printf("CONGRATS: The prescan function produced proper output.\n");
  }

  return 0;
}
