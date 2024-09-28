#include "compute.h"



// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
    // Check for valid input matrices and output pointer
    if (!a_matrix || !b_matrix || !output_matrix) {
        return -1;
    }

    // Calculate dimensions of the output matrix based on input matrices
    int out_rows = a_matrix->rows - b_matrix->rows + 1;
    int out_cols = a_matrix->cols - b_matrix->cols + 1;

    // Allocate memory for the output matrix structure
    *output_matrix = (matrix_t *)malloc(sizeof(matrix_t));
    if (!*output_matrix) {
        return -1;  // Return error if memory allocation fails
    }

    // Set dimensions for the output matrix
    (*output_matrix)->rows = out_rows;
    (*output_matrix)->cols = out_cols;
    // Allocate memory for the data in the output matrix
    (*output_matrix)->data = (int32_t *)malloc(out_rows * out_cols * sizeof(int32_t));
    if (!(*output_matrix)->data) {
        free(*output_matrix);  // Free the allocated structure in case of allocation failure
        return -1;  // Return error if memory allocation fails
    }

    // Loop over each element in the output matrix
    for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
            int sum = 0; // To store the convolution result for each element
            // Perform element-wise multiplication and summation
            for (int k = 0; k < b_matrix->rows; k++) {
                for (int l = 0; l < b_matrix->cols; l++) {
                    // Calculate the convolution sum by multiplying corresponding elements
                    // Note: b_matrix is flipped horizontally and vertically
                    sum += a_matrix->data[(i + k) * a_matrix->cols + (j + l)] * b_matrix->data[(b_matrix->rows - 1 - k) * b_matrix->cols + (b_matrix->cols - 1 - l)];
                }
            }
            // Store the sum in the output matrix
            (*output_matrix)->data[i * out_cols + j] = sum;
        }
    }

    return 0; // Return 0 to indicate successful completion
}

// Executes a task
int execute_task(task_t *task) {
  matrix_t *a_matrix, *b_matrix, *output_matrix;

  char *a_matrix_path = get_a_matrix_path(task);
  if (read_matrix(a_matrix_path, &a_matrix)) {
    printf("Error reading matrix from %s\n", a_matrix_path);
    return -1;
  }
  free(a_matrix_path);

  char *b_matrix_path = get_b_matrix_path(task);
  if (read_matrix(b_matrix_path, &b_matrix)) {
    printf("Error reading matrix from %s\n", b_matrix_path);
    return -1;
  }
  free(b_matrix_path);

  if (convolve(a_matrix, b_matrix, &output_matrix)) {
    printf("convolve returned a non-zero integer\n");
    return -1;
  }

  char *output_matrix_path = get_output_matrix_path(task);
  if (write_matrix(output_matrix_path, output_matrix)) {
    printf("Error writing matrix to %s\n", output_matrix_path);
    return -1;
  }
  free(output_matrix_path);

  free(a_matrix->data);
  free(b_matrix->data);
  free(output_matrix->data);
  free(a_matrix);
  free(b_matrix);
  free(output_matrix);
  return 0;
}
