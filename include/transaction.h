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

    // calls _initializeIdentifiers (so the logic won't be implemented in the constructor)
    Transaction(TransactionType type, const std::vector<int>& indices);

    // print the transaction
    void print() const;

    // return a string in format: "[type, indices[0], indices[1],...,indices[n-1]]".
    // E.g. for type='A' and indices={1,3,2}, identifier="[A,1,3,2]"
    static std::string createIdentifier(char type, const std::vector<int>& indices);

private:
    TransactionType _type;
    std::vector<int> _indices;

    //using the type and indices of the operation, generate all the identifiers (id, result, dependencies)
    //which will be used during Diekert graph calculation
    void _initializeIdentifiers();
};
