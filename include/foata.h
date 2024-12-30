#pragma once
#include "../include/graphs.h"
#include "../include/transaction.h"

// using the dependency graph, connect each Node in the Diekert graph
void addEdgesFromDependency(DiekertGraph& graph, const DependencyGraph& dependency);

// apply the transitive reduction to the Diekert graph
DiekertGraph reduceTransitively(DiekertGraph& graph, const std::vector<std::string>& word);

// create a Diekert graph from a word and a dependency graph
DiekertGraph createDiekertGraph(const std::vector<std::string>& word, const DependencyGraph& dependency);

// apply a modified BFS to find Foata levels for each Node in Diekert graph
std::vector<int> getFoataMaxPaths(DiekertGraph& diekert);

// get Foata Normal Form using the vector of levels
void getFoataForm(DiekertGraph& diekert, const std::vector<int>& maxPaths,
                const std::map<std::string, Transaction>& transactions,
                std::vector<std::vector<Transaction>>& foata);

// print Foata Normal Form
void printFoataForm(const std::vector<std::vector<Transaction>>& foata);

// create a dependency graph
DependencyGraph dependencyGraph(const std::map<std::string, Transaction>& transactions);

// create an independency graph
DependencyGraph independencyGraph(const DependencyGraph& dependency, const std::set<std::string>& alphabet);