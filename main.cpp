#include <iostream>
#include <set>

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

int main() {
    std::vector<Transaction> transactions;
    generateTransactions(transactions, 3);

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
