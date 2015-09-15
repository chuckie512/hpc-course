/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD
 * Student: 
 * Pthreads parallel Strassen algorithm for matrix multiplication.
 */

#include <cstdio>
#include <cstdlib>
#include "timer.h"
#include "io.h"
#include <pthread.h>
#include <cstdlib>
#include <ctime>

// Stupid simple Matrix Multiplication, meant as example.
void stupidMM(int N, int **A, int **B, int **C) {
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<N; k++) {
	C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

// Main method
int main(int argc, char* argv[]) {
  int N;
  int **A, **B, **C;
  double elapsedTime;

  // checking parameters
  if (argc != 2 && argc != 4) {
    cout << "Parameters: <N> [<fileA> <fileB>]" << endl;
    return 1;
  }
  N = atoi(argv[1]);

  // allocating matrices
  A = new int*[N];
  B = new int*[N];
  C = new int*[N];
  for (int i=0; i<N; i++){
    A[i] = new int[N];
    B[i] = new int[N];
    C[i] = new int[N];
  }

  // reading files (optional)
  if(argc == 4){
    readMatrixFile(A,N,argv[2]);
    readMatrixFile(B,N,argv[3]);
  } else {
    // Otherwise, generate random matrix.
    std::srand(std::time(0));
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
	A[i][j] = std::rand() % 10;
	B[i][j] = std::rand() % 10;
      }
    }
  }
  cout << "------- MATRIX A" << std::endl;
  printMatrix(A,N);
  cout << "------- MATRIX B" << std::endl;
  printMatrix(B,N);

  // starting timer
  timerStart();

  // YOUR CODE GOES HERE

  // testing the results is correct
  if(argc == 4){
    cout << "------- MATRIX C" << std::endl;
    printMatrix(C,N);
  }
  
  // stopping timer
  elapsedTime = timerStop();

  cout << "Duration: " << elapsedTime << " seconds" << std::endl;

  // releasing memory
  for (int i=0; i<N; i++) {
    delete [] A[i];
    delete [] B[i];
    delete [] C[i];
  }
  delete [] A;
  delete [] B;
  delete [] C;

  return 0;
}
