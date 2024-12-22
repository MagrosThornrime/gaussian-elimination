#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <map>

struct Node {
    std::string transactionID;
    std::vector<int> neighbors;
};

struct Graph {
    std::vector<Node> nodes;

    // save graph to a file in .dot format
    void saveAsDot(const std::string& filename) const;

    // print each Node and its neighbors
    void printNodes() const;
};

struct DependencyGraph : Graph {
    std::map<std::string, int> indexes;

    // print all edges in the graph
    void printEdges() const;

    // check if there exists a Node with given transaction id
    bool containsNode(const std::string& transactionID) const;

    // check if there exists an edge of given Nodes
    bool containsEdge(const std::string& key1, const std::string& key2) const;

    // add an empty Node with given transaction id
    void addNode(const std::string& transactionID);

    // add and edge between two given Nodes
    void addEdge(const std::string& key1, const std::string& key2);
};

struct DiekertGraph : Graph {
    // add an edge between two given Nodes
    void addEdge(int from, int to);

    // get number of Nodes
    int getSize() const;

    // get a reference to Node with given index
    Node& getNode(int index);

    // add an empty Node with given transaction id
    void addNode(const std::string& transactionID);

    // check if there is an edge of two given Nodes
    bool containsEdge(int from, int to) const;

    // create empty Nodes for each transaction in word
    void addNodesFromWord(const std::vector<std::string>& word);
};