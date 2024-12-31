#pragma once

#ifdef __CUDACC__
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif

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
    int i{}, j{}, k{};

    // check if at least one of two transactions depends on one another.
    static bool areDependent(const Transaction& first, const Transaction& second);

    // calls _initializeIdentifiers (so the logic won't be implemented in the constructor)
    Transaction(TransactionType type, const std::vector<int>& indices);

    // print the transaction
    void print() const;

    // return a string in format: "[type, indices[0], indices[1],...,indices[n-1]]".
    // E.g. for type='A' and indices={1,3,2}, identifier="[A,1,3,2]"
    static std::string createIdentifier(char type, const std::vector<int>& indices);

    // calculate the operation
    CUDA_DEV void calculate(double* matrix, double* multipliers, double* subtractors,
                            int matrixRowSize, int multipliersRowSize) const;

private:
    TransactionType _type;

    //using the type and indices of the operation, generate all the identifiers (id, result, dependencies)
    //which will be used during Diekert graph calculation
    void _initializeIdentifiers(const std::vector<int>& indices);
};
