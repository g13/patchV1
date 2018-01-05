// read inputs
// to do:
//  add function: read in predetermined input spike times
#ifndef INPUT_ARGS_H
#define INPUT_ARGS_H
#include <boost/program_options.hpp>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
// my naming convention, functions and methods "_", variables "caMel", types "CaMel"
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::ifstream;
using std::streampos;
using std::ios;

namespace po = boost::program_options;

struct input_args {
	po::options_description initialGenerics, cmdLineOptions, configFileOptions;
    po::variables_map vm;
    string theme;
    unsigned long nInput;
    unsigned long seed;
    vector<double> tstep, runTime, inputLevel;
    vector<vector<double>> input;
    string inputMode;
    bool irregInputLevels, dtVarLevels, tVarLevels;
    bool tVar, pVar, extendVar, withDend;
    unsigned long nNeuron;
    string configFn, inputLevelsFn, inputFn, conFn, strFn, reformatInputFn;
    vector<vector<unsigned long>> preID;
    vector<vector<unsigned long>> preStr;
    input_args();
    int read(int argc, char **argv);
    int reformat_input_table(double tstep0);
};
typedef struct input_args InputArgs;

template <typename T>
int read_input_table(string tableFn, vector<unsigned long> column, vector<vector<T>> &arr) {
    // arr are arrays needing to be fed.
    ifstream tableFile;
    tableFile.open(tableFn, ios::binary);
    cout << " reading from file " << tableFn << endl;
    int fSize;
    unsigned long length;
    if (tableFile) {
        tableFile.seekg(0,tableFile.end);
        fSize = tableFile.tellg();
        tableFile.seekg(0,tableFile.beg);
        cout << " file size: " << fSize << " bytes" << endl;
        if (column.size() == 1 && arr.size() == 0) {
            cout << "uniform columns of " << column[0] << " not knowing rows " << endl;
            int ii = 0;
            while (tableFile.tellg() < fSize) {
                arr.push_back(vector<T>(column[0],0));
                tableFile.read((char*)&(arr[ii][0]),sizeof(T)*column[0]);
                cout << ii << ": " << arr[ii][0];
                for (int j=1; j<column[0]; j++) {
                    cout << ", " << arr[ii][j]; 
                }
                cout << endl;
                ii = ii + 1;
            }
            assert(tableFile.tellg() == fSize);
        } else {
            cout << arr.size() << " rows"<< endl;
            if (column.size() == 0) {
                length = fSize/(sizeof(T)*arr.size());
                column.assign(arr.size(),length);
                cout <<  " derived uniform columns of " << length << endl;
            } else {
                if (column.size() == 1 && arr.size() > 1) {
                    column.assign(arr.size(),column[0]);
                    cout <<  " uniform " << length << " columns" << endl;
                }
            } // else non-uniform columns
            assert( column.size() == arr.size() );
            for (int i=0; i<arr.size(); i++) {
                arr[i].assign(column[i],0);
                tableFile.read((char*)&(arr[i][0]),sizeof(T)*column[i]);
                cout << i << ": " << arr[i][0];
                for (int j=1; j<column[i]; j++) {
                    cout << ", " << arr[i][j]; 
                }
                cout << endl;
            }
        }
    }
    tableFile.close();
    return 1;
}
#endif
