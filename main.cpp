#include <iostream>
#include <set>

#include "include/algorithm.h"

int main() {
    Transaction transaction;
    transaction.setValues("[A,1,2]", "[m,2,1]", {"[M,2,1]", "[M,1,1]"});
    transaction.print();
}
