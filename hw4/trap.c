/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD
 * Student: 
 * Implement Pthreads version of trapezoidal approximation.
 */

#include <stdio.h>
#include "timer.h"

// Global variables to make coverting to pthreads easier :)
double a;
double b;
int n;
double approx;

// Actual areas under the f(x) = x^2 curves, for you to check your
// values against.
double static NEG_1_TO_POS_1 = 0.66666666666667;
double static ZERO_TO_POS_10 = 333.333;

// f function is defined a x^2
double f(double a) {
  return a * a;
}

// Serial implementation of trapezoidal approximation. You should
// refactor the loop in this function to be parallized using pthread.
void trap() {
  double h = (b-a) / n;
  approx = ( f(a) - f(b) ) / 2.0;
  for(int i = 1; i < n-1; i++) {
    double x_i = a + i*h;
    approx += f(x_i);
  }
  approx = h*approx;
}

int main() {
  // Example 1 [-1,1]
  a = -1.0;
  b = 1.0;
  n = 1000000000;
  timerStart();
  trap();
  printf("Took %ld ms\n", timerStop());
  printf("a:%f\t b:%f\t n:%d\t actual:%f\t approximation:%f\n", a, b, n, NEG_1_TO_POS_1, approx);

  // Example 2 [0,10]
  a = 0.0;
  b = 10.0;
  n = 1000000000;
  timerStart();
  trap();
  printf("Took %ld ms\n", timerStop());
  printf("a:%f\t b:%f\t n:%d\t actual:%f\t approximation:%f\n", a, b, n, ZERO_TO_POS_10, approx);

  return 0;
}
