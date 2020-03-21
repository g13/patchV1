#ifndef UTIL_H
#define UTIL_H
#include <vector>
#include <utility>
#include <functional>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include "../types.h"

inline Float get_rand_from_gauss(Float p[], std::default_random_engine &rGen, std::function<bool(Float)> &outOfBound) {
    static std::normal_distribution<Float> norm(0.0, 1.0);
	Float v;
	//Size count = 0;
	//std::cout << "p0: " << p[0] << ", " << p[1] << "\n";
    do {
        Float rand = norm(rGen);
        v = p[0] + rand*p[1];
		//count++;
		//if (count > 10) {
		//	std::cout << count << ": " << rand << ", " << v << "\n";
		//}
		//if (count > 20) {
		//	assert(count <= 20);
		//}
    } while (outOfBound(v));
	return v;
}

inline std::pair<Float, Float> get_rands_from_correlated_gauss(Float p1[], Float p2[], Float rho, Float rho_comp, std::default_random_engine &rGen1, std::default_random_engine &rGen2, std::function<bool(Float)> &outOfBound1, std::function<bool(Float)> &outOfBound2) {
    static std::normal_distribution<Float> norm(0.0, 1.0);
    Float rand1, rand2, v1, v2;
	//Size count = 0;
	//std::cout << "p1: " << p1[0] << ", " << p1[1] << "\n";
	//std::cout << "p2: " << p2[0] << ", " << p2[1] << "\n";
    do {
        rand1 = norm(rGen1);
        v1 = p1[0] + rand1*p1[1];
		//if (count > 10) {
		//	std::cout  << count << ": " << rand1 << ", " << v1 << "\n";
		//}
		//if (count > 20) {
		//	assert(count <= 20);
		//}
    } while (outOfBound1(v1));
	//count = 0;
    do {
        rand2 = norm(rGen2);
        v2 = p2[0] + (rho*rand1 + rho_comp*rand2)*p2[1];
		//if (count > 10) {
		//	std::cout << count << ": " << rand2 << ", " << v2 << ", " << rand1 << "\n";
		//}
		//if (count > 20) {
		//	assert(count <= 20);
		//}
    } while (outOfBound2(v2));
    return std::make_pair(v1, v2);
}

template <typename T>
std::pair<T, T> lognstats(const T true_mean, const T true_std) {
	T log_mean = log(true_mean*true_mean / sqrt(true_std*true_std + true_mean * true_mean));
	T log_std = sqrt(log(true_std*true_std / (true_mean*true_mean) + 1));
	return std::make_pair(log_mean, log_std);
}

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
        input_file.read(reinterpret_cast<char*>(&listSize), sizeof(Size));
        if (!input_file) break;
		std::vector<T> new_data(listSize);
		if (listSize > 0) {
			input_file.read(reinterpret_cast<char*>(&new_data[0]), listSize * sizeof(T));
		}
		data.push_back(new_data);
    } while (true);
	input_file.close();
	return data;
} 

template <typename T>
void write_listOfListForArray(std::string filename, std::vector<std::vector<T>> data, bool append=false) {
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
    Size nList = data.size();
    output_file.write((char*)&nList, sizeof(Size));
    Size maxList = 0;
	for (Size i=0; i<nList; i++) {
        if (data[i].size() > maxList) {
            maxList = data[i].size();
        }
    }
    output_file.write((char*)&maxList, sizeof(Size));
	for (Size i=0; i<nList; i++) {
        Size listSize = data[i].size();
        output_file.write((char*)&listSize, sizeof(Size));
		if (listSize > 0) {
			output_file.write((char*)&data[i][0], listSize * sizeof(T));
		}
    }
	output_file.close();
}

template <typename T>
void read_listOfListToArray(std::string filename, T* &array, Size &maxList, bool print = false, T ratio = 1) {
	std::ifstream input_file;
	input_file.open(filename, std::fstream::in | std::fstream::binary);
	if (!input_file) {
		std::string errMsg{ "Cannot open or find " + filename + "\n" };
		throw errMsg;
	}
    Size nList;
    input_file.read(reinterpret_cast<char*>(&nList), sizeof(Size));
    input_file.read(reinterpret_cast<char*>(&maxList), sizeof(Size));
    size_t arraySize = nList*maxList;
    array = new T[arraySize];
    for (PosInt i=0; i<nList; i++) {
        Size listSize;
        input_file.read(reinterpret_cast<char*>(&listSize), sizeof(Size));
		input_file.read(reinterpret_cast<char*>(&array[i*maxList]), listSize * sizeof(T));
		if (ratio != 1) {
			for (PosInt j=0; j<listSize; j++) {
				array[i*maxList + j] *= ratio;
			}
		}
    }
	input_file.close();
} 

// denorm is the min number of advance made in u1 such that u1 and u2 has no phase difference.
inline
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

template <typename T>
std::pair<T, T> average2D(std::vector<T> x, std::vector<T> y) {
    Float mx = std::accumulate(x.begin(), x.end(), 0.0f)/x.size();
    Float my = std::accumulate(y.begin(), y.end(), 0.0f)/y.size();
    return std::make_pair(mx,my);
}

template <typename T>
std::pair<T, T> array_minmax(T* v, size_t n) {
    T min, max;
    assert(n>0);
    min = v[0];
    max = v[0];
    for (PosInt i=1; i<n; i++) {
        if (v[i] < min) {
            min = v[i];
        } else {
            if (v[i] > max) {
                max =  v[i];
            }
        }
    }
    return std::make_pair(min, max);
}

template <typename T>
T array_max(T* v, size_t n) {
    T max;
    assert(n>0);
    max = v[0];
    for (PosInt i=1; i<n; i++) {
        if (v[i] > max) {
            max =  v[i];
        }
    }
    return max;
}
#endif
