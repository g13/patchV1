#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
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

template <typename T>
void write_listOfList(std:string filename, std::vector<std::vector<T>> data, bool append=false) {
	std::ofstream output_file;
	if (append) {
		output_file.open(filename, std::fstream::out|std::fstream::app|std::fstream::binary);
	} else {
		output_file.open(filename, std::fstream::out|std::fstream::binary);
	}
	if (!output_file) {
		cout << "Cannot open or find " << filename <<"\n";
		return EXIT_FAILURE;
	}
	for (Size i=0; i<data.size(); i++) {
        Size listSize = data[i].size();
        output_file.write((char*)&listSize, sizeof(Size));
        output_file.write((char*)&data[i][0], listSize * sizeof(T));
    }
	output_file.close();
}

template <typename T>
std::vector<std::vector<T>> read_listOfList(std:string filename, bool print = false) {
	std::ifstream input_file;
	input_file.open(filename, std::fstream::in | std::fstream::binary);
	if (!input_file) {
		cout << "Cannot open or find " << filename <<"\n";
		return EXIT_FAILURE;
	}
	std::vector<std::vector<T>>	data;
	do {
        Size listSize;
        input_file.read(reinterpret_cast<char*>(&listSize), sizeof(Int));
		vector<T> new_data(listSize);
        input_file.read(reinterpret_cast<char*>(&new_data[0]), listSize * sizeof(D));
		data.push_back(new_data);
    } while (!input_file.eof());
	input_file.close()
	return data;
}
