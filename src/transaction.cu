#include "../include/transaction.cuh"
#include <sstream>
#include <iostream>

void Transaction::print() const {
    std::cout << "id: " << id << std::endl;
    std::cout << "result: " << result << std::endl;
    std::cout << "dependencies: ";
    for (const auto& dependency : dependencies) {
        std::cout << dependency << " ";
    }
    std::cout << std::endl;
}

bool Transaction::areDependent(const Transaction &first, const Transaction &second) {
    return first.dependencies.contains(second.result) || second.dependencies.contains(first.result);
}

Transaction::Transaction(TransactionType type, const std::vector<int>& indices) : _type(type){
    _initializeIdentifiers(indices);
}

std::string Transaction::createIdentifier(char type, const std::vector<int>& indices) {
    std::stringstream identifier;
    identifier << "[" << type;
    for (const auto index : indices) {
        identifier << "," << index;
    }
    identifier << "]";
    return identifier.str();
}

void Transaction::_initializeIdentifiers(const std::vector<int>& indices) {
    switch(_type) {
        case multiplier:
            if(indices.size() != 2) {
                throw std::invalid_argument("invalid number of indices");
            }
            i = indices[0];
            k = indices[1];
            id = createIdentifier('A', {i, k});
            result = createIdentifier('m', {k, i});
            dependencies.insert(createIdentifier('M', {k, i}));
            dependencies.insert(createIdentifier('M', {i, i}));
            break;
        case multiply:
            if(indices.size() != 3) {
                throw std::invalid_argument("invalid number of indices");
            }
            i = indices[0];
            j = indices[1];
            k = indices[2];
            id = createIdentifier('B', {i, j, k});
            result = createIdentifier('n', {i, j, k});
            dependencies.insert(createIdentifier('M', {i, j}));
            dependencies.insert(createIdentifier('m', {k, i}));
            break;
        case subtract:
            if(indices.size() != 3) {
                throw std::invalid_argument("invalid number of indices");
            }
            i = indices[0];
            j = indices[1];
            k = indices[2];
            id = createIdentifier('C', {i, j, k});
            result = createIdentifier('M', {k, j});
            dependencies.insert(createIdentifier('M', {k, j}));
            dependencies.insert(createIdentifier('n', {i, j, k}));
            break;
    }
}

CUDA_DEV void Transaction::calculate(double *matrix, double *multipliers, double *subtractors,
    int matrixRowSize, int multipliersRowSize) const {
    int subtractorIndex = i * matrixRowSize * multipliersRowSize + j * matrixRowSize + k;
    switch(_type) {
        case multiplier:
            multipliers[k * multipliersRowSize + i] = matrix[k * matrixRowSize + i] / matrix[i * matrixRowSize + i];
            break;
        case multiply:
            subtractors[subtractorIndex] = matrix[i * matrixRowSize + j] * multipliers[k * multipliersRowSize + i];
            break;
        case subtract:
            matrix[k * matrixRowSize + j] -= subtractors[subtractorIndex];
            break;
    }
}


