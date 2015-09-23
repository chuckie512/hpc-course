/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD
 * Student: Charles Smith <cas275@pitt.edu>
 * Implement Pthreads version of Strassen algorithm for matrix multiplication.
 */

#include "timer.h"
#include "io.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
int **allocMatrix(int size);
// Make these globals so threads can operate on them. You will need to
// add additional matrixes for all the M and C values in the Strassen
// algorithms.
int **A;
int **B;
int **C;
// Reference matrix, call simpleMM to populate.
int **R;

// Stupid simple Matrix Multiplication, meant as example.
void simpleMM(int N) {
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<N; k++) {
	R[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}


/* start my code */
// WRITE YOUR CODE HERE, you will need to also add functions for each
// of the sub-matrixes you will need to calculate but you can create your
// threads in this fucntion.

//because the matrix needs to be 2^n by 2^n

//function to check if n is a perfect square
//return 1 if true, else 0
int is_sq(int n){
    int temp = (int)sqrt(n*1.0);
    if (temp*temp==n){
        return 1;
    }
    return 0;
}

//this pads A with 0s so it is NxN where N is a perfect sq
void pad_0_A( int n){

    int ** m = A;
    if(is_sq(n))
       return;
    
    int half = (((int)sqrt(n*1.0))+1);
    int new_n = half*half;
    A = allocMatrix(new_n);
    for(int i = 0; i<n; i++)
        for(int j=0; j<n; j++)
            A[i][j]=m[i][j];
    
    free(m);

}

//this pads B with 0s 
void pad_0_B( int n){

    int ** m = B;
    if(is_sq(n))
       return m;

    int half = (((int)sqrt(n*1.0))+1);
    int new_n = half*half;

    B = allocMatrix(new_n);
    for(int i = 0; i<n; i++)
        for(int j=0; j<n; j++)
            B[i][j]=m[i][j];
    free(m);

}


int new_n(int n){
    if(is_sq(n))
        return n;
    int half = (((int)sqrt(n*1.0))+1);
    return half*half;
}



int ** m1;
int ** m2;
int ** m3;
int ** m4;
int ** m5;
int ** m6;
int ** m7;

void calc_m1(int n){
    //(A11+A22)(B11+B22)
    
    int ** a = allocMatrix(n/2);
    int ** b = allocMatrix(n/2); 
    for(int i=0; i<n/2; i++){
        for(int j=0; j<n/2; j++){
            a[i][j]=A[i][j]+A[n/2+i][n/2+j];
            b[i][j]=B[i][j]+B[n/2+i][n/2+j];
        }
    }
    for (int i=0; i<n/2; i++) {
         for (int j=0; j<n/2; j++) {
              for (int k=0; k<n/2; k++) {
                   m1[i][j] += a[i][k] * b[k][j];
              }
         }
    }
    free(a);
    free(b); 
}

void calc_m2(int n){
    //(A21+A22)(B11)

    int ** a = allocMatrix(n/2);
    for(int i=0; i<n/2; i++)
        for(int j=0; j<n/2; j++)
            a[i][j]=A[n/2+i][j]+A[n/2+i][n/2+j];
    for(int i=0; i<n/2; i++)
        for(int j=0; j<n/2; j++)
            for(int k=0; k<n/2; k++)
                m2[i][j]+=a[i][k]*B[k][j];
    free(a);
}

void calc_m3(int n){
    //(A11)(B12-B22)
    
    int ** b = allocMatrix(n/2);
    for(int i=0; i<n/2; i++)
        for(int j=0; j<n/2; j++)
            b[i][j]= B[i][n/2+j]-B[n/2+i][n/2+j];
    for(int i=0; i<n/2; i++)
        for(int j=0; j<n/2; j++)
            for(int k=0; k<n/2; k++)
                m3[i][j]+=A[i][k]*b[k][j];
    free(b);
}

void calc_m4(int n){
    //(A22)(B21-B11)

    int ** b = allocMatrix(n/2);
    for(int i=0; i<n/2;i++)
        for(int j=0; j<n/2; j++)
            b[i][j]=B[n/2+i][j]-B[i][j];
    for(int i=0; i<n/2; i++)
        for(int j=0; j<n/2; j++)
            for(int k=0; k<n/2; k++)
                m4[i][j]+=A[n/2+i][n/2+k]*b[k][j];
    free(b);
}

void calc_m5(int n){
    //(A11+A12)B22
 
    int ** a = allocMatrix(n/2);
    for(int i=0; i<n/2; i++)
        for(int j=0; j<n/2; j++)
            a[i][j]=A[i][j]+A[i][n/2+j];
    for(int i=0; i<n/2; i++)
        for(int j=0; j<n/2; j++)
            for(int k=0; k<n/2; k++)
                m5[i][j]+=a[i][k]*B[n/2+k][n/2+j];
    free(a);   
}

void calc_m6(int n){
    //(A21-A11)(B11+B12)

    int** a = allocMatrix(n/2);
    int** b = allocMatrix(n/2);
    for(int i=0; i<n/2; i++){
        for(int j=0; j<n/2; j++){
            a[i][j]=A[n/2+i][j]-A[i][j];
            b[i][j]=B[i][j]+B[i][n/2+j];
        }
    }
    for(int i=0; i<n/2; i++)
        for(int j=0; j<n/2; j++)
            for(int k=0; k<n/2; k++)
                m6[i][j]+=a[i][k]*b[k][j];

    free(a);
    free(b);

}

void calc_m7(int n){
    //(A12-A22)(B21+B22)
  
    int ** a = allocMatrix(n/2);
    int ** b = allocMatrix(n/2);

    for(int i=0; i<n/2; i++){
        for(int j=0; j<n/2; j++){
            a[i][j]=A[i][n/2+j]-A[n/2+i][n/2+j];
            b[i][j]=B[n/2+i][j]+B[n/2+i][n/2+j];
        }
    }
    for(int i=0; i<n/2; i++)
        for(int j=0; j<n/2; j++)
            for(int k=0; k<n/2; k++)
                m7[i][j]+=a[i][k]*b[k][j];
    free(a);
    free(b);
}


int ** c1;
int ** c2;
int ** c3;
int ** c4;

void calc_c1(int n){
    //m1+m4-m5+m7


    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            c1[i][j]=m1[i][j]+m4[i][j]-m5[i][j]+m7[i][j];
        }
    }

}

void calc_c2(int n){
    //m3+m5

    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            c2[i][j]=m3[i][j]+m5[i][j];
}

void calc_c3(int n){
    //m2+m4
    
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            c3[i][j]=m2[i][j]+m4[i][j];

}

void calc_c4(int n){
    //m1-m2+m3+m6
    //
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            c4[i][j]=(m1[i][j]-m2[i][j])+(m3[i][j]+m6[i][j]);

}


void strassenMM(int N) {
    pad_0_A(N);
    pad_0_B(N);
    
   
    
    
    int oldN = N;
    N = new_n(N);
 
    m1=allocMatrix(N/2);
    m2=allocMatrix(N/2);
    m3=allocMatrix(N/2);
    m4=allocMatrix(N/2);
    m5=allocMatrix(N/2);
    m6=allocMatrix(N/2);
    m7=allocMatrix(N/2);
    
    pthread_t m_threads[7];

    pthread_create(&m_threads[0],NULL, &calc_m1,(void*) N);
    pthread_create(&m_threads[1],NULL, &calc_m2,(void*) N);
    pthread_create(&m_threads[2],NULL, &calc_m3,(void*) N);
    pthread_create(&m_threads[3],NULL, &calc_m4,(void*) N);
    pthread_create(&m_threads[4],NULL, &calc_m5,(void*) N);
    pthread_create(&m_threads[5],NULL, &calc_m6,(void*) N);
    pthread_create(&m_threads[6],NULL, &calc_m7,(void*) N);
    for(int i=0; i<7; i++)
        pthread_join(m_threads[i], NULL);
    

    c1=allocMatrix(N/2);
    c2=allocMatrix(N/2);
    c3=allocMatrix(N/2);
    c4=allocMatrix(N/2);
    
    pthread_t c_threads[4];
    pthread_create(&c_threads[0],NULL,calc_c1,(void*)(N/2));
    pthread_create(&c_threads[1],NULL,calc_c2,(void*)(N/2));
    pthread_create(&c_threads[2],NULL,calc_c3,(void*)(N/2));
    pthread_create(&c_threads[3],NULL,calc_c4,(void*)(N/2));
    for(int i=0; i<4; i++)
        pthread_join(c_threads[i],NULL);


    free(m1);
    free(m2);
    free(m3);
    free(m4);
    free(m5);
    free(m6);
    free(m7);

    for(int i=0; i<oldN; i++){
        for(int j=0; j<oldN; j++){
            if(i<N/2&&j<N/2) //c1
                C[i][j]=c1[i][j];
            else if(i>=N/2&&j<N/2) //c3
                C[i][j]=c3[i-N/2][j];
            else if(i<N/2&&j>=N/2) //c2
                C[i][j]=c2[i][j-N/2];
            else if(i>=N/2&&j>=N/2) //c4
                C[i][j]=c4[i-N/2][j-N/2];

        }
    }

    free(c1);
    free(c2);
    free(c3);
    free(c4);
    
}

/* end my code */


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

// Allocate memory for all the matrixes, you will need to add code
// here to initialize any matrixes that you need.
void initMatrixes(int N) {
  A = allocMatrix(N); B = allocMatrix(N); C = allocMatrix(N); R = allocMatrix(N);
}

// Free up matrixes.
void cleanup() {
  free(A);
  free(B);
  free(C);
  free(R);
}

// Main method
int main(int argc, char* argv[]) {
  int N;
  double elapsedTime;

  // checking parameters
  if (argc != 2 && argc != 4) {
    printf("Parameters: <N> [<fileA> <fileB>]\n");
    return 1;
  }
  N = atoi(argv[1]);
  initMatrixes(N);

  // reading files (optional)
  if(argc == 4){
    readMatrixFile(A,N,argv[2]);
    readMatrixFile(B,N,argv[3]);
  } else {
    // Otherwise, generate two random matrix.
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
	A[i][j] = rand() % 5;
	B[i][j] = rand() % 5;
      }
    }
  }

  // Do simple multiplication and time it.
  timerStart();
  simpleMM(N);
  printf("Simple MM took %ld ms\n", timerStop());

  // Do strassen multiplication and time it.
  timerStart();
  strassenMM(N);
  printf("Strassen MM took %ld ms\n", timerStop());

  if (compareMatrix(C, R, N) != 0) {
    if (N < 20) {
      printf("\n\n------- MATRIX C\n");
      printMatrix(C,N);
      printf("\n------- MATRIX R\n");
      printMatrix(R,N);
    }
    printf("Matrix C doesn't match Matrix R, if N < 20 they will be printed above.\n");
  }

  // stopping timer
  
  cleanup();
  return 0;
}
