/*  Purpose: Calculate definite integral using trapezoidal rule.
 *
 * Input:   a, b, n, num_threads
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids, with num_threads.
 *
 * Compile: gcc -o trap trap.c -O3 -std=c99 -Wall -lpthread -lm
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 * Author: Naga Kandasamy
 * Date modified: 4/1/2019
 *
 */

#ifdef _WIN32
#  define NOMINMAX
#endif

// #define _REENTRANT // Make sure libraries are multi-threading-safe
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

typedef struct thread_params {
  int tid;
  int num_threads;
  int num_traps;
  float a;
  float b;
  float base_length;
  double *partial_integral;
} thread_params;

double compute_using_pthreads (float, float, int, float, int);
double compute_gold (float, float, int, float);
void *compute_integral(void *);
void print_args(thread_params);
int
main (int argc, char **argv)
{
    if (argc < 5) {
        printf ("Usage: trap lower-limit upper-limit num-trapezoids num-threads\n");
        printf ("lower-limit: The lower limit for the integral\n");
        printf ("upper-limit: The upper limit for the integral\n");
        printf ("num-trapezoids: Number of trapeziods used to approximate the area under the curve\n");
        printf ("num-threads: Number of threads to use in the calculation\n");
        exit (EXIT_FAILURE);
    }

    struct timeval start, stop;

    float a = atoi (argv[1]); /* Lower limit */
	float b = atof (argv[2]); /* Upper limit */
	float n = atof (argv[3]); /* Number of trapezoids */

	float h = (b - a)/(float) n; /* Base of each trapezoid */
	printf ("The base of the trapezoid is %f \n", h);

        gettimeofday(&start, NULL);
	double reference = compute_gold (a, b, n, h);
        gettimeofday(&stop, NULL);
    printf ("Reference solution computed using single-threaded version = %f \ntime = %0.2f s\n", reference, (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float) 1000000) );

	/* Write this function to complete the trapezoidal rule using pthreads. */
    int num_threads = atoi (argv[4]); /* Number of threads */


        gettimeofday(&start, NULL);
	double pthread_result = compute_using_pthreads (a, b, n, h, num_threads);
        gettimeofday(&stop, NULL);

	printf ("Solution computed using %d threads = %f \ntime = %0.2f\n", num_threads, pthread_result, (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float) 1000000) );

    exit(EXIT_SUCCESS);
}


/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Defines the integrand
 * Input args:  x
 * Output: sqrt((1 + x^2)/(1 + x^4))

 */
float
f (float x)
{
    return sqrt ((1 + x*x)/(1 + x*x*x*x));
}

/*------------------------------------------------------------------
 * Function:    compute_gold
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids using a single-threaded version
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral
 */
double
compute_gold (float a, float b, int n, float h)
{
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;

   for (k = 1; k <= n-1; k++)
     integral += f(a+k*h);

   integral = integral*h;

   return integral;
}

/* FIXME: Complete this function to perform the trapezoidal rule using pthreads. */
double
compute_using_pthreads (float a, float b, int n, float h, int num_threads)
{
    pthread_t *tids = (pthread_t *) malloc (num_threads * sizeof(pthread_t));
    pthread_attr_t attributes;
    pthread_attr_init (&attributes);

    int i;
    double *partial_integral = (double *) malloc (num_threads * sizeof(double));
    thread_params *params = (thread_params *) malloc (num_threads * sizeof(thread_params));

    for (i = 0; i < num_threads; i++) {
        params[i].tid = i;
        params[i].num_threads = num_threads;
        params[i].a = a;
        params[i].b = b;
        params[i].num_traps = n;
	params[i].base_length = h;
        params[i].partial_integral = partial_integral;
    }

    for (i = 0; i < num_threads; i++)
        pthread_create (&tids[i], &attributes, compute_integral, (void *) &params[i]);

    for (i = 0; i < num_threads; i++)
        pthread_join(tids[i], NULL);

    double integral = 0.0;

    integral = (f(a) + f(b))/2.0;
    for (i = 0; i < num_threads; i++)
        integral += partial_integral[i];

    free ((void *) params);
    free ((void *) partial_integral);

    return integral * h;
}

void *
compute_integral (void *args) {
    thread_params *params = (thread_params *) args;
    double partial_integral = 0.0;
    int i;

    for (i = 1 + params->tid; i <= params->num_traps - 1; i+=params->num_threads)
        partial_integral += f((params->a + i) * params->base_length);
    params->partial_integral[params->tid] = partial_integral;

    pthread_exit (NULL);
}

void
print_args (thread_params params) {
    printf("tid: %d\n", params.tid);
    printf("num_threads: %d\n", params.num_threads);
    printf("a: %d\n", params.a);
    printf("b: %d\n", params.b);
    printf("num_traps: %d\n", params.num_traps);
    printf("base_length: %f\n", params.base_length);
    printf("partial_integral: %lf\n", params.partial_integral[0]);
}
