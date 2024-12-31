#pragma once
#include <vector>

#include "transaction.cuh"

// find a multiplier for a matrix row, which will be used to subtract it from another row.
// the function is parallelized on the GPU
__global__ void findMultipliers(const double* matrix, const int* indices, double* multipliers, int matrixRowSize,
                                int indicesSize, int multipliersRowSize);

// multiply a row by the multiplier found before, then subtract it from another row.
// the function is parallelized on the GPU
__global__ void multiplyAndSubtractRows(double* matrix, const int* indices, const double* multipliers, int matrixRowSize,
                                        int indicesSize, int multipliersRowSize);

// perform an array of Transactions, calling .calculate() on each
// the function is parallelized on the GPU
__global__ void performTransactions(double* matrix, double* multipliers, double* subtractors, int matrixRowSize,
                                    int multipliersRowSize, const Transaction* transactions, int transactionsSize);

// transform the matrix into an upper triangular one using the Gaussian Elimination
void calculateGaussianElimination(std::vector<double>& matrix, int rows, int columns);

// transform the matrix into an upper triangular one using the Gaussian Elimination with Foata Normal Form
void calculateFoataElimination(std::vector<double>& matrix, int rows, int columns,
                                    const std::vector<std::vector<Transaction>>& foata);

// transform the upper triangular matrix calculated before to a singular matrix
void transformIntoSingular(std::vector<double>& matrix, int rows, int column);

// save matrix into file
void saveMatrix(const std::vector<double>& matrix, int rows, int columns, const std::string& path);

// read matrix from a file, return its number of rows
int readMatrix(std::vector<double>& matrix, const std::string& path);

// generate a list of transactions needed to calculate the Gaussian Elimination
void generateTransactions(std::vector<Transaction>& transactions, int matrixSize);