/* Code for the Jacbi equation solver. 
 * 
 * Author: Naga Kandasamy
 * Date created: April 26, 2019
 * Date modified: 
 *
 * Compile as follows:
 * gcc -fopenmp -o solver solver.c solver_gold.c -O3 -Wall -std=c99 -lm
 *
 * If you wish to see debug info add the -D DEBUG option when compiling the code.
 */

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "grid.h" 
#include <sys/time.h>

extern int compute_gold (grid_t *);
int compute_using_omp_jacobi (grid_t *, int);
void compute_grid_differences(grid_t *, grid_t *);
grid_t *create_grid (int, float, float);
grid_t *copy_grid (grid_t *);
void print_grid (grid_t *);
void print_stats (grid_t *);
double grid_mse (grid_t *, grid_t *);


int 
main (int argc, char **argv)
{	
	if (argc < 5) {
        printf ("Usage: %s grid-dimension num-threads min-temp max-temp\n", argv[0]);
        printf ("grid-dimension: The dimension of the grid\n");
        printf ("num-threads: Number of threads\n"); 
        printf ("min-temp, max-temp: Heat applied to the north side of the plate is uniformly distributed between min-temp and max-temp\n");
        exit (EXIT_FAILURE);
    }
    
    /* Parse command-line arguments. */
    int dim = atoi (argv[1]);
    int num_threads = atoi (argv[2]);
    float min_temp = atof (argv[3]);
    float max_temp = atof (argv[4]);
    struct timeval start, stop;
    /* Generate the grids and populate them with initial conditions. */
 	grid_t *grid_1 = create_grid (dim, min_temp, max_temp);
    /* Grid 2 should have the same initial conditions as Grid 1. */
    grid_t *grid_2 = copy_grid (grid_1); 

	/* Compute the reference solution using the single-threaded version. */
	printf ("\nUsing the single threaded version to solve the grid\n");
	gettimeofday(&start, NULL);
	int num_iter = compute_gold (grid_1);
	gettimeofday(&stop, NULL);
	printf ("Convergence achieved after %d iterations\n", num_iter);
    /* Print key statistics for the converged values. */
	printf ("Printing statistics for the interior grid points\n");
	float stime = stop.tv_sec - start.tv_sec + ((stop.tv_usec - start.tv_usec) / (float) 1000000);
	//printf ("Serial time: %0.2f seconds\n", (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float) 1000000) );
	printf ("Serial time: %0.2f seconds\n", stime);
    print_stats (grid_1);
#ifdef DEBUG
    print_grid (grid_1);
#endif
	
	/* Use omp to solve the equation using the jacobi method. */
	printf ("\nUsing omp to solve the grid using the jacobi method\n");
	gettimeofday(&start, NULL);
	num_iter = compute_using_omp_jacobi (grid_2, num_threads);
	gettimeofday(&stop, NULL);
	printf ("Convergence achieved after %d iterations\n", num_iter);			
    printf ("Printing statistics for the interior grid points\n");
	printf ("Parallel time: %0.2f seconds\n", (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float) 1000000) );
	printf ("Serial time: %0.2f seconds\n", stime);
	print_stats (grid_2);
#ifdef DEBUG
    print_grid (grid_2);
#endif
    
    /* Compute grid differences. */
    double mse = grid_mse (grid_1, grid_2);
    printf ("MSE between the two grids: %f\n", mse);

	/* Free up the grid data structures. */
	free ((void *) grid_1->element);	
	free ((void *) grid_1); 
	free ((void *) grid_2->element);	
	free ((void *) grid_2);

	exit (EXIT_SUCCESS);
}

/* FIXME: Edit this function to use the jacobi method of solving the equation. The final result should be placed in the grid data structure. */
int 
compute_using_omp_jacobi (grid_t *grid, int num_threads)
{
	int num_iter = 0;
        int done = 0;
    	int i, j;
        double diff;
        float old, new;
    	float eps = 1e-2; /* Convergence criteria. */
    	int num_elements = (grid->dim - 2) * (grid->dim - 2);
	grid_t *grid_copy = copy_grid(grid);
	grid_t *temp;

	while(!done) { /* While we have not converged yet. */
		#pragma omp parallel num_threads(num_threads) private(new, old)  shared(grid, grid_copy, temp, diff)
		{
			double local_diff = 0.0;
			#pragma omp for schedule(static)
			for (i = 1; i < (grid->dim - 1); i++) {
				for (j = 1; j < (grid->dim - 1); j++) {
					old = grid->element[i * grid->dim + j]; /* Store old value of grid point. */

					new = 0.25 * (grid->element[(i - 1) * grid->dim + j] +\
					grid->element[(i + 1) * grid->dim + j] +\
					grid->element[i * grid->dim + (j + 1)] +\
					grid->element[i * grid->dim + (j - 1)]);
					
					grid_copy->element[i * grid->dim + j] = new; /* Update the grid-point value. */
					local_diff = local_diff + fabs(new - old); /* Calculate the difference in values. */
				}
			}
			
			#pragma omp critical
			diff += local_diff;		


			temp = grid;
			grid = grid_copy;
			grid_copy = temp;
		}

		diff = diff / num_elements;
		/* End of an iteration. Check for convergence. */
		printf ("Iteration %d. DIFF: %f.\n", num_iter, diff);
		num_iter++;

		if (diff < eps)
			done = 1;
        }

	

    return num_iter;
}



/* Create a grid with the specified initial conditions. */
grid_t * 
create_grid (int dim, float min, float max)
{
    grid_t *grid = (grid_t *) malloc (sizeof (grid_t));
    if (grid == NULL)
        return NULL;

    grid->dim = dim;
	printf("Creating a grid of dimension %d x %d\n", grid->dim, grid->dim);
	grid->element = (float *) malloc (sizeof (float) * grid->dim * grid->dim);
    if (grid->element == NULL)
        return NULL;

    int i, j;
	for (i = 0; i < grid->dim; i++) {
		for (j = 0; j < grid->dim; j++) {
            grid->element[i * grid->dim + j] = 0.0; 			
		}
    }

    /* Initialize the north side, that is row 0, with temperature values. */ 
    srand ((unsigned) time (NULL));
	float val;		
    for (j = 1; j < (grid->dim - 1); j++) {
        val =  min + (max - min) * rand ()/(float)RAND_MAX;
        grid->element[j] = val; 	
    }

    return grid;
}

/* Creates a new grid and copies over the contents of an existing grid into it. */
grid_t *
copy_grid (grid_t *grid) 
{
    grid_t *new_grid = (grid_t *) malloc (sizeof (grid_t));
    if (new_grid == NULL)
        return NULL;

    new_grid->dim = grid->dim;
	new_grid->element = (float *) malloc (sizeof (float) * new_grid->dim * new_grid->dim);
    if (new_grid->element == NULL)
        return NULL;

    int i, j;
	for (i = 0; i < new_grid->dim; i++) {
		for (j = 0; j < new_grid->dim; j++) {
            new_grid->element[i * new_grid->dim + j] = grid->element[i * new_grid->dim + j] ; 			
		}
    }

    return new_grid;
}

/* This function prints the grid on the screen. */
void 
print_grid (grid_t *grid)
{
    int i, j;
    for (i = 0; i < grid->dim; i++) {
        for (j = 0; j < grid->dim; j++) {
            printf ("%f\t", grid->element[i * grid->dim + j]);
        }
        printf ("\n");
    }
    printf ("\n");
}


/* Print out statistics for the converged values of the interior grid points, including min, max, and average. */
void 
print_stats (grid_t *grid)
{
    float min = INFINITY;
    float max = 0.0;
    double sum = 0.0;
    int num_elem = 0;
    int i, j;

    for (i = 1; i < (grid->dim - 1); i++) {
        for (j = 1; j < (grid->dim - 1); j++) {
            sum += grid->element[i * grid->dim + j];

            if (grid->element[i * grid->dim + j] > max) 
                max = grid->element[i * grid->dim + j];

             if(grid->element[i * grid->dim + j] < min) 
                min = grid->element[i * grid->dim + j];
             
             num_elem++;
        }
    }
                    
    printf("AVG: %f\n", sum/num_elem);
	printf("MIN: %f\n", min);
	printf("MAX: %f\n", max);
	printf("\n");
}

/* Calculate the mean squared error between elements of two grids. */
double
grid_mse (grid_t *grid_1, grid_t *grid_2)
{
    double mse = 0.0;
    int num_elem = grid_1->dim * grid_1->dim;
    int i;

    for (i = 0; i < num_elem; i++) 
        mse += (grid_1->element[i] - grid_2->element[i]) * (grid_1->element[i] - grid_2->element[i]);
                   
    return mse/num_elem; 
}



		

