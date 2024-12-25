#pragma once
#include <fstream>
#include <set>
#include <vector>

enum TransactionType {
    multiplier, multiply, subtract
};

struct Transaction {
    std::string id;
    std::string result;
    std::set<std::string> dependencies;

    // check if at least one of two transactions depends on one another.
    static bool areDependent(const Transaction& first, const Transaction& second);

    Transaction(TransactionType type, const std::vector<int>& indices);
    // print the transaction
    void print() const;

    static std::string createIdentifier(char type, const std::vector<int>& indices);

private:
    TransactionType _type;
    std::vector<int> _indices;

    void _initializeIdentifiers();
};
