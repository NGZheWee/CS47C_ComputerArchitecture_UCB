#include <omp.h>
#include <x86intrin.h>

#include "compute.h"

static int32_t conv1d_align16(int32_t *arr0, int32_t *arr1, uint32_t size) {
  int32_t sum = 0;
  #pragma omp parallel for if (size >= 32) reduction(+: sum)
  for (uint32_t i = 0; i < size; i += 16) {
    __m256i a00 = _mm256_loadu_si256((__m256i *)(arr0 + i));
    __m256i a01 = _mm256_loadu_si256((__m256i *)(arr0 + i + 8));
    __m256i a10 = _mm256_loadu_si256((__m256i *)(arr1 + i));
    __m256i a11 = _mm256_loadu_si256((__m256i *)(arr1 + i + 8));
    __m256i mul0 = _mm256_mullo_epi32(a00, a10);
    __m256i mul1 = _mm256_mullo_epi32(a01, a11);
    __m256i sumv = _mm256_add_epi32(mul0, mul1);
    sumv = _mm256_hadd_epi32(sumv, sumv);
    sumv = _mm256_hadd_epi32(sumv, sumv);
    int32_t temp_sum = _mm_cvtsi128_si32(_mm_add_epi32(_mm256_castsi256_si128(sumv), _mm256_extracti128_si256(sumv, 1)));
    sum += temp_sum;
  }

  return sum;
}

static int32_t conv1d_w8(int32_t *arr0, int32_t *arr1) {
  int32_t temp[8];
  int32_t sum = 0;
  __m256i a00 = _mm256_loadu_si256((__m256i *)(arr0));
  __m256i a10 = _mm256_loadu_si256((__m256i *)(arr1));
  _mm256_storeu_si256((__m256i *)temp, _mm256_mullo_epi32(a00, a10));
  for (int i = 0; i < 8; i++) sum += temp[i];
  return sum;
}

static int32_t conv1d_w4(int32_t *arr0, int32_t *arr1) {
  int32_t temp[4];
  int32_t sum = 0;
  __m128i a00 = _mm_loadu_si128((__m128i *)(arr0));
  __m128i a10 = _mm_loadu_si128((__m128i *)(arr1));
  _mm_storeu_si128((__m128i *)temp, _mm_mullo_epi32(a00, a10));
  for (int i = 0; i < 4; i++) sum += temp[i];
  return sum;
}

static void conv1d_w8_1(int32_t* dst, int32_t *arr0, __m256i val) {
  _mm256_storeu_si256((__m256i *)dst, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i *)(arr0)), val));
}


static void conv1d_w4_1(int32_t* dst, int32_t *arr0, __m128i val) {
  _mm_storeu_si128((__m128i *)dst, _mm_mullo_epi32(_mm_loadu_si128((__m128i *)(arr0)), val));
}

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
    return -1; // Return error if memory allocation fails
  }

  // Set dimensions for the output matrix
  (*output_matrix)->rows = out_rows;
  (*output_matrix)->cols = out_cols;
  // Allocate memory for the data in the output matrix
  (*output_matrix)->data = (int32_t *)malloc(((out_rows * out_cols + 7) & ~7) * sizeof(int32_t));
  if (!(*output_matrix)->data) {
    free(*output_matrix); // Free the allocated structure in case of allocation failure
    return -1;            // Return error if memory allocation fails
  }

  int max_threads = omp_get_max_threads();
  uint32_t asize = a_matrix->rows * a_matrix->cols;
  uint32_t bsize = b_matrix->rows * b_matrix->cols;
  uint32_t ksize_align16 = ((bsize + 15) & ~15);
  int32_t *b_data = (int32_t *)malloc(sizeof(int32_t) * ksize_align16);
  memset(b_data, 0, sizeof(int32_t) * ksize_align16);
  int32_t **a_data = (int32_t **)malloc(sizeof(int32_t *) * max_threads);
  if (bsize > 1) {
    for (int i = 0; i < max_threads; i++) {
      a_data[i] = (int32_t *)malloc(sizeof(int32_t) * ksize_align16);
      memset(a_data[i], 0, sizeof(int32_t) * ksize_align16);
    }
  }

  // flip
  #pragma omp parallel for
  for (int k = 0; k < b_matrix->rows * b_matrix->cols; k++) {
    int i = k / b_matrix->cols;
    int j = k % b_matrix->cols;
    // Calculate the convolution sum by multiplying corresponding elements
    // Note: b_matrix is flipped horizontally and vertically
    b_data[j + i * b_matrix->cols] = b_matrix->data[(b_matrix->rows - 1 - i) * b_matrix->cols + (b_matrix->cols - 1 - j)];
  }

  if (bsize > 8) {
    // Loop over each element in the output matrix
    #pragma omp parallel for if (out_cols * out_rows >= max_threads)
    for (int k = 0; k < out_cols * out_rows; k++) {
      int i = k / out_cols;
      int j = k % out_cols;
      int tid = omp_get_thread_num();
      for (int l = 0; l < b_matrix->rows; l++) {
        memcpy(a_data[tid] + l * b_matrix->cols, a_matrix->data + (i + l) * a_matrix->cols + j, sizeof(int32_t) * b_matrix->cols);
      }
      // Store the sum in the output matrix
      (*output_matrix)->data[k] = conv1d_align16(a_data[tid], b_data, ksize_align16);
    }
  } else if (bsize > 4) {
    // Loop over each element in the output matrix
    #pragma omp parallel for
    for (int k = 0; k < out_cols * out_rows; k++) {
      int i = k / out_cols;
      int j = k % out_cols;
      int tid = omp_get_thread_num();
      for (int l = 0; l < b_matrix->rows; l++) {
        memcpy(a_data[tid] + l * b_matrix->cols, a_matrix->data + (i + l) * a_matrix->cols + j, sizeof(int32_t) * b_matrix->cols);
      }
      // Store the sum in the output matrix
      (*output_matrix)->data[k] = conv1d_w8(a_data[tid], b_data);
    }
  } else if (bsize > 1) {
    // Loop over each element in the output matrix
    #pragma omp parallel for
    for (int k = 0; k < out_cols * out_rows; k++) {
      int i = k / out_cols;
      int j = k % out_cols;
      int tid = omp_get_thread_num();
      for (int k = 0; k < b_matrix->rows; k++) {
        memcpy(a_data[tid] + k * b_matrix->cols, a_matrix->data + (i + k) * a_matrix->cols + j, sizeof(int32_t) * b_matrix->cols);
      }
      // Store the sum in the output matrix
      (*output_matrix)->data[k] = conv1d_w4(a_data[tid], b_data);
    }
  } else {
    if (asize >= 8) {
      __m256i val = _mm256_set1_epi32(b_data[0]);
      if (asize & 7) {
        conv1d_w8_1((*output_matrix)->data, a_matrix->data, val);
      }
      #pragma omp parallel for
      for (int i = asize & 7; i < asize; i += 8) {
        conv1d_w8_1((*output_matrix)->data + i, a_matrix->data + i, val);
      }

    } else if (asize >= 4) {
      __m128i val = _mm_set1_epi32(b_data[0]);
      if (asize & 3) {
        conv1d_w4_1((*output_matrix)->data , a_matrix->data, val);
      }
      conv1d_w4_1((*output_matrix)->data + (asize & 3), a_matrix->data + (asize & 3), val);

    } else if (asize > 1) {
      int32_t mask[2][4] = {
        {-1, -1,  0,  0},
        {-1, -1, -1,  0},
      };
      __m128i mask_v = _mm_loadu_si128((const __m128i *)(mask[asize - 2]));
      __m128i val = _mm_set1_epi32(b_data[0]);
      __m128i a = _mm_maskload_epi32(a_matrix->data, mask_v);
      _mm_maskstore_epi32((*output_matrix)->data, mask_v, _mm_mullo_epi32(a, val));
    } else {
      (*output_matrix)->data[0] = a_matrix->data[0] * b_data[0];
    }
  }
  if (bsize > 1) {
    for (int i = 0; i < max_threads; i++) {
      if (a_data[i]) {
        free(a_data[i]);
      }
    }
  }
  free(a_data);
  free(b_data);
  return 0;
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

