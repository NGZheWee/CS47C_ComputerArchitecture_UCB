#include "ex2.h"

double dotp_naive(double* x, double* y, int arr_size) {
    double global_sum = 0.0;
    for (int i = 0; i < arr_size; i++)
        global_sum += x[i] * y[i];
    return global_sum;
}

// Critical Keyword
double dotp_critical(double* x, double* y, int arr_size) {
    double global_sum = 0.0;
    // TODO: Implement this function

    // Start a parallel region. Each thread executes the code within this block.
    #pragma omp parallel
    {
        double local_sum = 0.0; // Define a local sum for each thread.

        // Distribute the iterations of the loop across the threads.
        #pragma omp for
        for (int i = 0; i < arr_size; i++) {
            local_sum += x[i] * y[i]; // Each thread computes part of the dot product.
        }

        // Only one thread at a time can execute the code within the critical section.
        #pragma omp critical
        {
            global_sum += local_sum; // Safely update the global sum with the local sum from each thread.
        }
    }
    // Use the critical keyword here!
    return global_sum;
}

// Reduction Keyword
double dotp_reduction(double* x, double* y, int arr_size) {
    double global_sum = 0.0;
    // TODO: Implement this function

    // Start a parallel loop with the reduction clause for addition on global_sum.
    // This allows each thread to have a local copy of global_sum, which are then
    // combined (reduced) at the end of the parallel region.
    #pragma omp parallel for reduction(+:global_sum)
    for (int i = 0; i < arr_size; i++) {
        global_sum += x[i] * y[i]; // Each thread contributes to the global sum directly.
    }


    // Use the reduction keyword here!
    return global_sum;
}

// Manual Reduction
double dotp_manual_reduction(double* x, double* y, int arr_size) {
    double global_sum = 0.0;
    // TODO: Implement this function
    
    // Start a parallel region. Each thread executes the code within this block.
    #pragma omp parallel
    {
        double local_sum = 0.0; // Define a local sum for each thread.

        // Distribute the iterations of the loop across the threads.
        #pragma omp for
        for (int i = 0; i < arr_size; i++) {
            local_sum += x[i] * y[i]; // Each thread computes part of the dot product.
        }

        // Only one thread at a time can execute the code within the critical section.
        #pragma omp critical
        {
            global_sum += local_sum; // Safely update the global sum with the local sum from each thread.
        }
    }
    // Do NOT use the `reduction` directive here!
    return global_sum;
}

