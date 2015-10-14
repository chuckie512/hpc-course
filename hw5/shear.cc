/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD (bmills@cs.pitt.edu)
 * STUDENTS: CHARLES SMITH
 * Implement OpenMP parallel shear sort.
 */

#include <omp.h>
#include <math.h>
#include "timer.h"
#include "io.h"

#define MAX_VALUE 10000

//MY STUFF

void sort_row(int **A, int M, int r){ //who doesn't like a nice bubble sort?
  for(int i =0; i<M; i++){
    for(int j=0; j<M-1; j++){
      if(A[r][j]>A[r][j+1]){
        int temp = A[r][j];
        A[r][j]=A[r][j+1];
        A[r][j+1]=temp;
      }
    }
  }
}

void sort_row_rev(int **A, int M, int r){
  for(int i =0; i<M; i++){
    for(int j=0; j<M-1; j++){
      if(A[r][j]<A[r][j+1]){
        int temp = A[r][j];
        A[r][j]=A[r][j+1];
        A[r][j+1]=temp;
      }
    }
  }
}

void sort_col(int **A, int M, int c){
  for(int j=0;j<M;j++){
    for(int k=0;k<M-1;k++){
      if(A[k][c]>A[k+1][c]){
        int temp=A[k][c];
        A[k][c]=A[k+1][c];
        A[k+1][c]=temp;

      }
    }
  }
}


void shear_sort(int **A, int M) {
  // Students: Implement parallel shear sort here.
  for(int i=0; i<log(M*M); i++){//outer loop DO NOT PARALLEL 

#pragma parallel for
    for(int k=0; k<M; k++){
      if(k%2==0){
        sort_row(A,M,k);
      }
      else{
        sort_row_rev(A,M,k); 
      }
    }  

#pragma parallel for
    for(int q=0; q<M; q++){
      sort_col(A,M,q);
    }

 
  } //end outer for

}

//END


// Allocate square matrix.
int **allocMatrix(int size) {
  int **matrix;
  matrix = (int **)malloc(size * sizeof(int *));
  for (int row = 0; row < size; row++) {
    matrix[row] = (int *)malloc(size * sizeof(int));
  }
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      matrix[i][j] = 0;
    }
  }
  return matrix;
}

// Main method      
int main(int argc, char* argv[]) {
  int N, M;
  int **A;
  double elapsedTime;

  // checking parameters
  if (argc != 2 && argc != 3) {
    printf("Parameters: <N> [<file>]\n");
    return 1;
  }
  N = atoi(argv[1]);
  M = (int) sqrt(N); 
  if(N != M*M){
    printf("N has to be a perfect square!\n");
    exit(1);
  }

  // allocating matrix A
  A = allocMatrix(M);

  // reading files (optional)
  if(argc == 3){
    readMatrixFile(A,M,argv[2]);
  } else {
    srand (time(NULL));
    // Otherwise, generate random matrix.
    for (int i=0; i<M; i++) {
      for (int j=0; j<M; j++) {
	A[i][j] = rand() % MAX_VALUE;
      }
    }
  }
  
  // starting timer
  timerStart();

  // calling shear sort function
  shear_sort(A,M);
  // stopping timer
  elapsedTime = timerStop();

  // print if reasonably small
  if (M <= 10) {
    printMatrix(A,M);
  }

  printf("Took %ld ms\n", timerStop());

  // releasing memory
  for (int i=0; i<M; i++) {
    delete [] A[i];
  }
  delete [] A;

  return 0;
}
