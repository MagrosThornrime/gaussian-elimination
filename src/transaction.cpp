#include "../include/transaction.h"
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

Transaction::Transaction(TransactionType type, const std::vector<int>& indices) : _type(type), _indices(indices) {
    _initializeIdentifiers();
}

std::string Transaction::createIdentifier(char type, const std::vector<int>& indices) {
    std::stringstream identifier;
    identifier << "[" << type;
    for (const auto index : indices) {
        identifier << "," << index + 1;
    }
    identifier << "]";
    return identifier.str();
}

void Transaction::_initializeIdentifiers() {
    switch(_type) {
        case multiplier:
            if(_indices.size() != 2) {
                throw std::invalid_argument("invalid number of indices");
            }
            id = createIdentifier('A', {_indices[0], _indices[1]});
            result = createIdentifier('m', {_indices[1], _indices[0]});
            dependencies.insert(createIdentifier('M', {_indices[1], _indices[0]}));
            dependencies.insert(createIdentifier('M', {_indices[0], _indices[0]}));
            break;
        case multiply:
            if(_indices.size() != 3) {
                throw std::invalid_argument("invalid number of indices");
            }
            id = createIdentifier('B', {_indices[0], _indices[1], _indices[2]});
            result = createIdentifier('n', {_indices[0], _indices[1], _indices[2]});
            dependencies.insert(createIdentifier('M', {_indices[0], _indices[1]}));
            dependencies.insert(createIdentifier('m', {_indices[2], _indices[0]}));
            break;
        case subtract:
            if(_indices.size() != 3) {
                throw std::invalid_argument("invalid number of indices");
            }
            id = createIdentifier('C', {_indices[0], _indices[1], _indices[2]});
            result = createIdentifier('M', {_indices[2], _indices[1]});
            dependencies.insert(createIdentifier('M', {_indices[2], _indices[1]}));
            dependencies.insert(createIdentifier('n', {_indices[0], _indices[1], _indices[2]}));
            break;
    }
}

