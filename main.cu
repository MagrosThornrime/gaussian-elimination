#include <iostream>
#include <set>
#include <chrono>

#include "include/foata.h"
#include "include/elimination.cuh"

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

void testFoataElimination(const std::string& inputFile, const std::string& outputFile, const std::string& graphOutputFile) {
    std::vector<double> matrix;
    const int rows = readMatrix(matrix, inputFile);
    const int columns = rows + 1;

    std::vector<Transaction> transactions;
    generateTransactions(transactions, rows);

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

    std::cout << "Word = (";
    for(const auto& identifier : word) {
        std::cout << identifier << ", ";
    }
    std::cout << ")" << std::endl;

    std::cout << "D = ";
    auto dependency = dependencyGraph(transactionsMapped);
    dependency.printEdges();

    const std::set alphabet(word.begin(), word.end());
    auto independency = independencyGraph(dependency, alphabet);
    std::cout << "I = ";
    independency.printEdges();

    auto diekert = createDiekertGraph(word, dependency);
    auto foataMaxPaths = getFoataMaxPaths(diekert);

    std::vector<std::vector<Transaction>> foata;
    getFoataForm(diekert, foataMaxPaths, transactionsMapped, foata);
    std::cout << "FNF = ";
    printFoataForm(foata);
    std::cout << std::endl;

    diekert.saveAsDot(graphOutputFile);

    calculateFoataElimination(matrix, rows, columns, foata);
    transformIntoSingular(matrix, rows, columns);
    saveMatrix(matrix, rows, columns, outputFile);
}

void testGaussianElimination(const std::string& inputFile, const std::string& outputFile) {
    std::vector<double> matrix;
    const int rows = readMatrix(matrix, inputFile);
    const int columns = rows + 1;
    calculateGaussianElimination(matrix, rows, columns);
    transformIntoSingular(matrix, rows, columns);
    saveMatrix(matrix, rows, columns, outputFile);
}

int main(){
    auto start = std::chrono::high_resolution_clock::now();
    testFoataElimination("../input.txt", "../output-foata.txt", "../graph.dot");
    auto foata = std::chrono::high_resolution_clock::now();
    testGaussianElimination("../input.txt", "../output-gaussian.txt");
    auto gaussian = std::chrono::high_resolution_clock::now();
    auto foataDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(foata - start);
    auto gaussianDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(gaussian - foata);
    std::cout << "\"Foata Elimination\" execution time: " << foataDuration.count() << " ns" << std::endl;
    std::cout << "Gaussian Elimination execution time: " << gaussianDuration.count() << " ns" << std::endl;

}