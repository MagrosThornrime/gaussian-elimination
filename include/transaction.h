#pragma once
#include <fstream>
#include <set>
#include <map>

struct Transaction {
    std::string id;
    std::string result;
    std::set<std::string> dependencies;

    // check if at least one of two transactions depends on one another.
    static bool areDependent(const Transaction& first, const Transaction& second);

    void setValues(const std::string& id, const std::string& result, const std::set<std::string>& dependencies);

    // print the transaction
    void print() const;
};
