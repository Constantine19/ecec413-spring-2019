/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date created: February 7
 * Date of last update: April 10, 2019
 *
 * Compile as follows: gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -std=c99 -lpthread -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define num_threads 16
#define num_elements MATRIX_SIZE

/* Function prototypes. */
extern int compute_gold (float *, unsigned int);
Matrix allocate_matrix (int, int, int);
void gauss_eliminate_using_pthreads (float *);
void *row_reduction (void *);
void *elimination (void *);
int perform_simple_check (const Matrix);
void print_matrix (const Matrix);
float get_random_number (int, int);
int check_results (float *, float *, unsigned int, float);

struct arr_entry 
{
    int elements;
    int id;
    float* mat;
};

// void* reduce_row(void *s);

int
main (int argc, char **argv)
{
    /* Check command line arguments. */
    if (argc > 1) {
        printf ("Error. This program accepts no arguments.\n");
        exit (EXIT_FAILURE);
    }

    Matrix A;			    /* Input matrix. */
    Matrix U_reference;		/* Upper triangular matrix computed by reference code. */
    Matrix U_mt;			/* Upper triangular matrix computed by pthreads. */

    /* Initialize the random number generator with a seed value. */
    srand (time (NULL));

    A = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 1);	            /* Allocate and populate a random square matrix. */
    U_reference = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	/* Allocate space for the reference result. */
    U_mt = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	        /* Allocate space for the multi-threaded result. */

    /* Copy the contents of the A matrix into the U matrices. */
    for (int i = 0; i < A.num_rows; i++) {
        for (int j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    printf ("Performing gaussian elimination using the reference code.\n");
    struct timeval start, stop;
    gettimeofday (&start, NULL);
    
    int status = compute_gold (U_reference.elements, A.num_rows);
  
    gettimeofday (&stop, NULL);
    printf ("CPU run time = %0.2f s.\n", (float) (stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec) / (float) 1000000));

    if (status == 0) {
        printf ("Failed to convert given matrix to upper triangular. Try again. Exiting.\n");
        exit (EXIT_FAILURE);
    }
  
    status = perform_simple_check (U_reference);	/* Check that the principal diagonal elements are 1. */ 
    if (status == 0) {
        printf ("The upper triangular matrix is incorrect. Exiting.\n");
        exit (EXIT_FAILURE);
    }
    printf ("Single-threaded Gaussian elimination was successful.\n");
  
    /* Perform the Gaussian elimination using pthreads. The resulting upper 
     * triangular matrix should be returned in U_mt */
    gettimeofday (&start, NULL);
    gauss_eliminate_using_pthreads(U_mt.elements);
    gettimeofday (&stop, NULL);
    printf ("PThreads CPU run time = %0.4f s. \n",
    (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float) 1000000));

    /* check if the pthread result matches the reference solution within a specified tolerance. */
    int size = MATRIX_SIZE * MATRIX_SIZE;
    int res = check_results (U_reference.elements, U_mt.elements, size, 0.0001f);
    printf ("TEST %s\n", (1 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for the matrices. */
    free (A.elements);
    free (U_reference.elements);
    free (U_mt.elements);

    exit (EXIT_SUCCESS);
}


/* FIXME: Write code to perform gaussian elimination using pthreads. */
void
gauss_eliminate_using_pthreads (float *U_mt)
{
    unsigned int elements;

    for (elements = 0; elements < num_elements; elements++) // perform Gaussian elimination
    {
        pthread_t thread_arr[num_threads];
        struct arr_entry* entry_point = malloc(num_threads * sizeof(struct arr_entry));

        unsigned int i, j, k, m;
        for (i = 0; i < num_threads; i++)
        {
            entry_point[i].elements = elements;
            entry_point[i].id = i;
            entry_point[i].mat = U_mt;

            pthread_create(&thread_arr[i], NULL, row_reduction, (void *)&entry_point[i]);
        }
    
        for (j = 0; j < num_threads; j++)
        {
            pthread_join(thread_arr[j], NULL);
        }

        U_mt[num_elements * elements + elements] = 1;

        for (k = 0; k < num_threads; k++)
        {
            entry_point[k].elements = elements;
            entry_point[k].id = k;
            entry_point[k].mat = U_mt;

            pthread_create(&thread_arr[k], NULL, elimination, (void *)&entry_point[k]);
        }

        for (m = 0; m < num_threads; m++)
        {
            pthread_join(thread_arr[m], NULL);
        }

        free(entry_point);
    }
}


void *row_reduction(void *s)
{
    unsigned int idx_r;
    struct arr_entry* myStruct = (struct arr_entry*) s;
    int elements = myStruct->elements;
    int id = myStruct->id;
    float* U_mt = myStruct->mat;

    for (idx_r = elements+id+1; idx_r < num_elements;)
    {
         /* Chunking */
        float num = U_mt[num_elements * elements + idx_r];
        float denom = U_mt[num_elements * elements + elements];
        float div_step = num / denom;
        U_mt[num_elements * elements + idx_r] = div_step;
        
        idx_r = idx_r + num_threads;
    }

    pthread_exit(0);
}


void *elimination(void *s)
{
    struct arr_entry* myStruct = (struct arr_entry*) s;
    int elements = myStruct->elements;
    int id = myStruct->id;
    float* U_mt = myStruct->mat;
    unsigned int  idx_el_1, idx_el_2;

    for (idx_el_1 = (elements + id)+1; idx_el_1 < num_elements; )
    {
        for (idx_el_2 = elements+1; idx_el_2 < num_elements; idx_el_2++)
        {
            float first_part = U_mt[num_elements * idx_el_1 + idx_el_2];
            float last_part = (U_mt[num_elements * idx_el_1 + elements] * U_mt[num_elements * elements + idx_el_2]);
            float elim_step = first_part - last_part;
            U_mt[num_elements * idx_el_1 + idx_el_2] = elim_step;
        }
    
        U_mt[num_elements * idx_el_1 + elements] = 0;
        idx_el_1 = idx_el_1 + num_threads;
  }
  pthread_exit(0);
}

/* Function checks if the results generated by the single threaded and multi threaded versions match. */
int
check_results (float *A, float *B, unsigned int size, float tolerance)
{
    for (int i = 0; i < size; i++)
        if (fabsf (A[i] - B[i]) > tolerance)
            return 0;
    return 1;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix
allocate_matrix (int num_rows, int num_columns, int init)
{
    Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *) malloc (size * sizeof (float));
  
    for (unsigned int i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
    }
  
    return M;
}


/* Returns a random floating-point number between the specified min and max values. */ 
float
get_random_number (int min, int max)
{
    return (float)
        floor ((double)
                (min + (max - min + 1) * ((float) rand () / (float) RAND_MAX)));
}

/* Performs a simple check on the upper triangular matrix. Checks to see if the principal diagonal elements are 1. */
int
perform_simple_check (const Matrix M)
{
    for (unsigned int i = 0; i < M.num_rows; i++)
        if ((fabs (M.elements[M.num_rows * i + i] - 1.0)) > 0.0001)
            return 0;
  
    return 1;
}
