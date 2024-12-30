#include <iostream>
#include <set>

#include "include/foata.h"
#include "include/elimination.cuh"

const std::string GRAPH_OUTPUT = "../graph.dot";
const std::string INPUT = "../input.txt";
const std::string OUTPUT = "../output.txt";

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

    std::vector<std::vector<Transaction>> foata;
    getFoataForm(diekert, foataMaxPaths, transactionsMapped, foata);
    std::cout << "FNF = ";
    printFoataForm(foata);
    std::cout << std::endl;


    diekert.saveAsDot(GRAPH_OUTPUT);
}

void testGaussianElimination() {
    std::vector<double> matrix;
    const int rows = readMatrix(matrix, INPUT);
    const int columns = rows + 1;
    calculateGaussianElimination(matrix, rows, columns);
    transformIntoSingular(matrix, rows, columns);
    saveMatrix(matrix, rows, columns, OUTPUT);
}

int main(){
    int rows = 4;
    calculateFoata(rows);
    // testGaussianElimination();
}