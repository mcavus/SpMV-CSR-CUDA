//
//  serial.c
//  CS 426 - Project 4
//
//  Created by Muhammed Cavusoglu on 25.05.2019.
//  Copyright © 2019 Muhammed Cavusoglu. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void read_matrix(int **row_ptr, int **col_ind, float **values, const char *filename, int *num_rows, int *num_cols, int *num_vals);

// Serial SpMV using CSR format
void spmv_csr(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const float *x, float *y) {
    for (int i = 0; i < num_rows; i++) {
        float dotProduct = 0;
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            dotProduct += values[j] * x[col_ind[j]];
        }
        
        y[i] = dotProduct;
    }
}

int main(int argc, const char * argv[]) {
    if (argc != 4) {
        fprintf(stdout, "Invalid command, enter:\n1. number of repetitions, 2. print mode (1 or 2), 3. test filename\n");
        exit(0);
    }
    
    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values, elapsed_time;;
    
    int num_repeat = atoi(argv[1]);
    int print_mode = atoi(argv[2]);
    const char *filename = argv[3];
    
    read_matrix(&row_ptr, &col_ind, &values, filename, &num_rows, &num_cols, &num_vals);
    
    float *x = (float *) malloc(num_rows * sizeof(float));
    float *y = (float *) malloc(num_rows * sizeof(float));
    for (int i = 0; i < num_rows; i++) {
        x[i] = 1.0;
        y[i] = 0.0;
    }
    
    if (print_mode == 1) {
        // Values Array
        fprintf(stdout, "Values Array:\n");
        for (int i = 0; i < num_vals; i++) {
            fprintf(stdout, "%.6f ", values[i]);
        }
        
        // Column Indices Array
        fprintf(stdout, "\n\nColumn Indices Array:\n");
        for (int i = 0; i < num_vals; i++) {
            fprintf(stdout, "%d ", col_ind[i]);
        }
        
        // Row Pointer Array
        fprintf(stdout, "\n\nRow Pointer Array:\n");
        for (int i = 0; i < (num_rows + 1); i++) {
            fprintf(stdout, "%d ", row_ptr[i]);
        }
        
        fprintf(stdout, "\n\nInitial Vector:\n");
        for (int i = 0; i < num_rows; i++) {
            fprintf(stdout, "%.1f ", x[i]);
        }
        
        fprintf(stdout, "\n\nResulting Vector:\n");
    }
    
    // Time the iterations
    clock_t start = clock();
    for (int i = 0; i < num_repeat; i++) {
        spmv_csr(row_ptr, col_ind, values, num_rows, x, y);
        
        // Copy the result to x_{i} at the end of each iteration, and use it in iteration x_{i+1}
        for (int i = 0; i < num_rows; i++) {
            x[i] = y[i];
            y[i] = 0.0;
        }
    }
    clock_t stop = clock();
    elapsed_time = (((float) (stop - start)) / CLOCKS_PER_SEC) * 1000; // in milliseconds

    // Print resulting vector
    if (print_mode == 1 || print_mode == 2) {
        for (int i = 0; i < num_rows; i++) {
            fprintf(stdout, "%.6f ", x[i]);
        }
        fprintf(stdout, "\n");
    }
    
    // Print elapsed time
    printf("\nSerial Running time:  %.4f ms\n", elapsed_time);
    
    free(row_ptr);
    free(col_ind);
    free(values);
    
    return 0;
}

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(int **row_ptr, int **col_ind, float **values, const char *filename, int *num_rows, int *num_cols, int *num_vals) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    
    // Get number of rows, columns, and non-zero values
    fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals);
    
    int *row_ptr_t = (int *) malloc((*num_rows + 1) * sizeof(int));
    int *col_ind_t = (int *) malloc(*num_vals * sizeof(int));
    float *values_t = (float *) malloc(*num_vals * sizeof(float));
    
    // Collect occurances of each row for determining the indices of row_ptr
    int *row_occurances = (int *) malloc(*num_rows * sizeof(int));
    for (int i = 0; i < *num_rows; i++) {
        row_occurances[i] = 0;
    }
    
    int row, column;
    float value;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;
        
        row_occurances[row]++;
    }
    
    // Set row_ptr
    int index = 0;
    for (int i = 0; i < *num_rows; i++) {
        row_ptr_t[i] = index;
        index += row_occurances[i];
    }
    row_ptr_t[*num_rows] = *num_vals;
    free(row_occurances);
    
    // Set the file position to the beginning of the file
    rewind(file);
    
    // Read the file again, save column indices and values
    for (int i = 0; i < *num_vals; i++) {
        col_ind_t[i] = -1;
    }
    
    fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals);
    int i = 0;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
        row--;
        column--;
        
        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1) {
            i++;
        }
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        i = 0;
    }
    
    fclose(file);
    
    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;
}
