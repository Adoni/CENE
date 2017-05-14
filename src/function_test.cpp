//
// Created by Adoni1203 on 16/7/19.
//
#include <iostream>
#include <vector>

using namespace std;

int main() {
  vector<int> a(10, 17);
  for (auto i : a) {
    cout << i << endl;
  }
  return 0;
}
