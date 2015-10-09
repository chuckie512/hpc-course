/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD (bmills@cs.pitt.edu)
 * Students:
 * Implement openmp verions of conway's game of life.
 */

#include "timer.h"
#include "io.h"

// Function implementing Conway's Game of Life
void conway(int **World, int N, int M){
  // STUDENT: IMPLEMENT THE GAME HERE, make it parallel!
}

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
  int N,M;
  int **World;
  double elapsedTime;

  // checking parameters
  if (argc != 3 && argc != 4) {
    printf("Parameters: <N> <M> [<file>]\n");
    return 1;
  }
  N = atoi(argv[1]);
  M = atoi(argv[2]);

  // allocating matrices
  World = allocMatrix(N);

  // reading files (optional)
  if(argc == 4){
    readMatrixFile(World,N,argv[3]);
  } else {
    // Otherwise, generate two random matrix.
    srand (time(NULL));
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
	World[i][j] = rand() % 2;
      }
    }
  }

  // starting timer
  timerStart();

  // calling conway's game of life 
  conway(World,N,M);

  // stopping timer
  elapsedTime = timerStop();

  printMatrix(World,N);

  printf("Took %ld ms\n", timerStop());

  // releasing memory
  for (int i=0; i<N; i++) {
    delete [] World[i];
  }
  delete [] World;

  return 0;
}
