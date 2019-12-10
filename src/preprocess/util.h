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
void write_listOfList(std::string filename, std::vector<std::vector<T>> data, bool append=false) {
	std::ofstream output_file;
	if (append) {
		output_file.open(filename, std::fstream::out|std::fstream::app|std::fstream::binary);
	} else {
		output_file.open(filename, std::fstream::out|std::fstream::binary);
	}
	if (!output_file) {
		std::string errMsg{ "Cannot open or find " + filename + "\n" };
		throw errMsg;
	}
	for (Size i=0; i<data.size(); i++) {
        Size listSize = data[i].size();
        output_file.write((char*)&listSize, sizeof(Size));
		if (listSize > 0) {
			output_file.write((char*)&data[i][0], listSize * sizeof(T));
		}
    }
	output_file.close();
}

template <typename T>
std::vector<std::vector<T>> read_listOfList(std::string filename, bool print = false) {
	std::ifstream input_file;
	input_file.open(filename, std::fstream::in | std::fstream::binary);
	if (!input_file) {
		std::string errMsg{ "Cannot open or find " + filename + "\n" };
		throw errMsg;
	}
	std::vector<std::vector<T>>	data;
	do {
        Size listSize;
        input_file.read(reinterpret_cast<char*>(&listSize), sizeof(Int));
		std::vector<T> new_data(listSize);
		if (listSize > 0) {
			input_file.read(reinterpret_cast<char*>(&new_data[0]), listSize * sizeof(T));
		}
		data.push_back(new_data);
    } while (!input_file.eof());
	input_file.close();
	return data;
} 

// denorm is the min number of advance made in u1 such that u1 and u2 has no phase difference.
PosInt find_denorm(PosInt u1, PosInt u2, bool MorN, PosInt &norm) { 
    PosInt m, n;
    if (MorN) { //u2 > u1
        m = u1;
        if (u2%u1 == 0) { // only one zero phase
            norm = 1;
            return 1;
        }
        n = u2 - u1*(u2/u1);
    } else {
        m = u2;
        if (u1%u2 == 0) { // only one zero phase
            norm = 1;
            return 1;
        }
        n = u1 - u2*(u1/u2);
    }
    printf("m = %u, n = %u\n", m, n);
    assert (m>n);
    for (PosInt i=n; i>1; i--) {
        if (n%i==0 && m%i==0) { 
            norm = n/i;
            return m/i;
        } 
    }
    norm = 1;
    return m;
} 

auto average(std::vector<Float> x, std::vector<Float> y) {
    Float mx = std::accumulate(x.begin(), x.end(), 0.0f)/x.size();
    Float my = std::accumulate(y.begin(), y.end(), 0.0f)/y.size();
    return make_pair(mx,my);
}

