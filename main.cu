#include <iostream>
#include <set>
#include <iomanip>

#include "include/algorithm.h"

const std::string OUTPUT_FILENAME = "../graph.dot";

void generateTransactions(std::vector<Transaction>& transactions, int matrixSize) {
    for(int i=0; i<matrixSize-1; i++) {
        for(int k=i+1; k<matrixSize; k++) {
            auto multiplier = Transaction(TransactionType::multiplier, {i, k});
            transactions.push_back(multiplier);
            for(int j=i; j<matrixSize+1; j++) {
                auto multiply = Transaction(TransactionType::multiply, {i, j, k});
                transactions.push_back(multiply);
                auto subtract = Transaction(TransactionType::subtract, {i, j, k});
                transactions.push_back(subtract);
            }
        }
    }
}

void calculateFoata(int matrixSize) {
    std::vector<Transaction> transactions;
    generateTransactions(transactions, matrixSize);

    std::map<std::string, Transaction> transactionsMapped;
    std::vector<std::string> word;

    for(const auto& transaction : transactions) {
        transactionsMapped.insert({transaction.id, transaction});
        word.push_back(transaction.id);
    }

    std::cout << "Alphabet = {";
    for(const auto& identifier : word) {
        std::cout << identifier << ", ";
    }
    std::cout << "}" << std::endl;

    std::cout << "D = ";
    auto dependency = dependencyGraph(transactionsMapped);
    dependency.printEdges();

    const std::set alphabet(word.begin(), word.end());
    auto independency = independencyGraph(dependency, alphabet);
    std::cout << "I = ";
    independency.printEdges();

    auto diekert = createDiekertGraph(word, dependency);
    auto foataMaxPaths = getFoataMaxPaths(diekert);
    std::cout << "FNF = ";
    printFoataForm(diekert, foataMaxPaths);

    diekert.saveAsDot(OUTPUT_FILENAME);
}

__global__ void findMultiplier(const double* matrix, const int* indices, double* multipliers, int matrixRowSize,
                                int indicesSize, int multipliersRowSize) {
    const int currentIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(currentIndex >= indicesSize) {
        return;
    }
    const int i = indices[2 * currentIndex];
    const int k = indices[2 * currentIndex + 1];
    multipliers[k * multipliersRowSize + i] = matrix[k * matrixRowSize + i] / matrix[i * matrixRowSize + i];
}

__global__ void multiplyAndSubtractRow(double* matrix, const int* indices, const double* multipliers, int matrixRowSize,
                                        int indicesSize, int multipliersRowSize) {
    const int currentIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(currentIndex >= indicesSize) {
        return;
    }
    const int i = indices[3 * currentIndex];
    const int j = indices[3 * currentIndex + 1];
    const int k = indices[3 * currentIndex + 2];
    matrix[k * matrixRowSize + j] -= matrix[i * matrixRowSize + j] * multipliers[k *multipliersRowSize + i];
}

void generateRandomMatrix(std::vector<double>& matrix, int rows, int columns, double minValue, double maxValue) {
    for(int i=0; i<rows * (columns + 1); i++) {
        matrix.push_back(minValue + rand() / (RAND_MAX/(maxValue-minValue)));
    }
}

void printMatrix(const std::vector<double>& matrix, int rows, int columns) {
    std::cout << rows << std::endl;
    for(int i=0; i<rows; i++) {
        for(int j=0; j<columns-1; j++) {
            std::cout << std::setprecision(16) << std::fixed <<matrix[i * columns + j] << " ";
        }
        std::cout<<std::endl;
    }
    for(int i=0; i<rows; i++) {
        std::cout<<std::setprecision(16) << std::fixed<<matrix[i*columns + columns - 1] << " ";
    }
    std::cout<<std::endl << std::endl;
}

void calculateGaussianElimination(std::vector<double>& matrix, int rows, int columns) {
    double* cudaMatrix = nullptr;
    cudaMalloc((void**)&cudaMatrix, rows * columns * sizeof(double));
    cudaMemcpy(cudaMatrix, matrix.data(), rows * columns * sizeof(double), cudaMemcpyHostToDevice);

    std::vector multipliers(rows * rows, 0.0);
    double* cudaMultipliers = nullptr;
    cudaMalloc((void**)&cudaMultipliers, rows * rows * sizeof(double));

    for(int i=0; i<rows-1; i++) {
        std::vector<int> indicesMultiplier;
        std::vector<int> indicesSubtract;
        for(int k=i+1; k<rows; k++) {
            indicesMultiplier.push_back(i);
            indicesMultiplier.push_back(k);
            for(int j=i; j<columns; j++) {
                indicesSubtract.push_back(i);
                indicesSubtract.push_back(j);
                indicesSubtract.push_back(k);
            }
        }
        cudaMemcpy(cudaMultipliers, multipliers.data(), rows * rows * sizeof(double), cudaMemcpyHostToDevice);

        int* cudaIndicesMultiplier = nullptr;
        cudaMalloc((void**)&cudaIndicesMultiplier, indicesMultiplier.size() * sizeof(int));
        cudaMemcpy(cudaIndicesMultiplier, indicesMultiplier.data(), indicesMultiplier.size() * sizeof(int), cudaMemcpyHostToDevice);

        int blocks = indicesMultiplier.size() / 1024 + 1;
        findMultiplier<<<blocks, 1024>>>(cudaMatrix, cudaIndicesMultiplier, cudaMultipliers,
                                        columns, indicesMultiplier.size() / 2, rows);

        int* cudaIndicesSubtract = nullptr;
        cudaMalloc((void**)&cudaIndicesSubtract, indicesSubtract.size() * sizeof(int));
        cudaMemcpy(cudaIndicesSubtract, indicesSubtract.data(), indicesSubtract.size() * sizeof(int), cudaMemcpyHostToDevice);

        blocks = indicesSubtract.size() / 1024 + 1;
        multiplyAndSubtractRow<<<blocks, 1024>>>(cudaMatrix, cudaIndicesSubtract, cudaMultipliers,  columns,
                        indicesSubtract.size() / 3, rows);

        cudaFree(cudaIndicesMultiplier);
        cudaFree(cudaIndicesSubtract);
    }

    cudaMemcpy(matrix.data(), cudaMatrix, rows * columns * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(cudaMultipliers);
    cudaFree(cudaMatrix);
}

void transformIntoSingular(std::vector<double>& matrix, int rows, int columns) {
    for(int i=rows-1; i>=0; i--) {
        matrix[i * columns + columns - 1] /= matrix[i * columns + i];
        matrix[i * columns + i] = 1.0;
        for(int j=i-1; j>=0; j--) {
            matrix[j * columns + columns - 1] -= matrix[j * columns + i] * matrix[i * columns + columns - 1];
            matrix[j * columns + i] = 0.0;
        }
    }
}



int main(){
    const int rows = 10;
    const int columns = rows + 1;
    std::vector<double> matrix;
    generateRandomMatrix(matrix, rows,  columns, 0.0, 40.0);
    printMatrix(matrix, rows, columns);
    calculateGaussianElimination(matrix, rows, columns);
    transformIntoSingular(matrix, rows, columns);
    printMatrix(matrix, rows, columns);
}