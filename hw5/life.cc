/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD (bmills@cs.pitt.edu)
 * Student: Charles Smith
 * Implement openmp verions of conway's game of life.
 */

#include "timer.h"
#include "io.h"

/*
 * int **World is the matrix
 * int N is the dimentions
 * int i is the row
 * int j is the column
 * returns the number of adjacent entries
 *
 * *** THIS RUNS IN SERIAL ***
 *
 */
int num_adjacent(int **world, int N, int i, int j){
  int count = 0;
  int x =i;
  int y =j;
  //check up
  x=i-1;
  y=j;
  if(x>=0&&world[x][y]==1)
    count++;
  //check UR
  x=i-1;
  y=j+1;
  if(x>=0&&y<N&&world[x][y]==1)
    count++;
  //check R
  x=i;
  y=j+1;
  if(y<N&&world[x][y]==1)
    count++;
  //check DR
  x=i+1;
  y=j+1;
  if(x<N&&y<N&&world[x][y]==1)
    count++;
  //check D
  x=i+1;
  y=j;
  if(x<N&&world[x][y]==1)
    count++;
  //check DL
  x=i+1;
  y=j-1;
  if(x<N&&y>=0&&world[x][y]==1)
    count++;
  //check L
  x=i;
  y=j-1;
  if(y>=0&&world[x][y]==1)
    count++;
  //check UL
  x=i-1;
  y=j-1;
  if(x>=0&&y>=0&&world[x][y]==1)
    count++;

  //return
  return count;
}

// Function implementing Conway's Game of Life
void conway(int **World, int N, int M){
  // STUDENT: IMPLEMENT THE GAME HERE, make it parallel!
  //  *** This is serial for now ***
  //  *** fix this later         **
   
  for(int x =0; x< M; x++){     //outer loop, this iterates through the different steps of the game. ** DO NOT PARALLEL **
    for(int i =0; i<N; i++){    //iterates over the rows
      for(int j =0; j<N; j++){  //iterates through the columns
        int count = num_adjacent(World, N, i, j);
        //printf("%d ",count);
        if     (count < 2) //loneliness
          World[i][j] = 0;
        else if(count > 3) //crowding
          World[i][j] = 0;
        else if(count ==3) //reproduction 
          World[i][j] = 1;
        //else survival
        
      } //end j (column loop)
      //printf("\n");
    }   //end i (row    loop)
  }     //end x (step   loop)
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
