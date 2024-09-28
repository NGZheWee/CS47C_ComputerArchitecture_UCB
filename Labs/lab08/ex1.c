#include "ex1.h"

void v_add_naive(double* x, double* y, double* z) {
    #pragma omp parallel
    {
        for(int i=0; i<ARRAY_SIZE; i++)
            z[i] = x[i] + y[i];
    }
}

// Adjacent Method
void v_add_optimized_adjacent(double* x, double* y, double* z) {
    // TODO: Implement this function
    #pragma omp parallel
    {
        // Get the total number of threads in the current parallel region.
        int num_threads = omp_get_num_threads();
        // Get the ID (or number) of the current thread.
        int thread_id = omp_get_thread_num();

        // Iterate over the array, with each thread handling elements that are spaced
        // 'num_threads' apart. This ensures that the i-th element handled by a thread
        // is congruent to the thread's ID modulo the total number of threads.
        for (int i = thread_id; i < ARRAY_SIZE; i += num_threads) {
            // Perform the vector addition for the element.
            z[i] = x[i] + y[i];
        }
    }
    // Do NOT use the `for` directive here!
}

// Chunks Method
void v_add_optimized_chunks(double* x, double* y, double* z) {
    // TODO: Implement this function
    #pragma omp parallel
    {
        // Get the total number of threads in the current parallel region.
        int num_threads = omp_get_num_threads();
        // Get the ID (or number) of the current thread.
        int thread_id = omp_get_thread_num();

        // Calculate the size of each chunk. Each thread will handle approximately
        // 'chunk_size' elements.
        int chunk_size = ARRAY_SIZE / num_threads;
        // Calculate the starting index for each thread.
        int start = thread_id * chunk_size;
        // Calculate the ending index. The last thread may have a larger chunk
        // if ARRAY_SIZE is not evenly divisible by the number of threads.
        int end = (thread_id == num_threads - 1) ? ARRAY_SIZE : start + chunk_size;

        // Iterate over the array chunk assigned to the current thread.
        for (int i = start; i < end; i++) {
            // Perform the vector addition for each element in the chunk.
            z[i] = x[i] + y[i];
        }
    }
    // Do NOT use the `for` directive here!
}


