#include <vector>
#include <utility>
#include <iostream>
#include <string>
#include <cassert>

template <typename T1, typename T2>
void print_pair(std::pair<std::vector<T1>, std::vector<T2>> coord) {
	for (Size i = 0; i < coord.first.size(); i++) {
		std::cout << "(" << coord.first[i] << "," << coord.second[i] << ")";
		if (i != coord.first.size() - 1) {
			std::cout << ", ";
		}
		else {
			std::cout << "\n";
		}
	}
}

template <typename T>
void print_list(std::vector<T> x, std::string begin = "\0", std::string end = "\0") {
	if (begin != "\0") {
		std::cout << begin;
	}
	for (Size i = 0; i < x.size(); i++) {
		std::cout << x[i];
		if (i != x.size() - 1) {
			std::cout << ", ";
		}
	}
	if (end != "\0") {
		std::cout << end;
	}
	std::cout << "\n";
}

template <typename T>
void print_listOfList(std::vector<std::vector<T>> x, std::string begin = "{", std::string end = "}") {
	for (Size i = 0; i < x.size(); i++) {
		print_list(x[i], begin, end);
	}
}

template <typename T>
void print_matrix(std::vector<std::vector<T>> mat, std::string begin = "| ", std::string end = " |") {
	const Size nrow = mat.size();
	const Size ncolumn = mat[0].size();
	for (Size i = 0; i < nrow; i++) {
		assert(mat[i].size() == ncolumn);
		print_list(mat[i], begin, end);
	}
}
