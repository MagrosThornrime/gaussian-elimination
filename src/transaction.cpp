#include "../include/transaction.h"
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

void Transaction::setValues(const std::string &id, const std::string &result, const std::set<std::string> &dependencies) {
    this->id = id;
    this->result = result;
    this->dependencies = dependencies;
}


