#include "input_args.h"
#include <cstdlib>
#include <cmath>
#include <ctime>
using std::ifstream;
using std::ofstream;
using std::ios;

input_args::input_args() {
    initialGenerics.add_options()
        ("configFn,c", po::value<string>(&configFn)->default_value("input.cfg"),"filename of configuration");
    configFileOptions.add_options()
        ("theme,m", po::value<string>(&theme), "theme of simulation")
        ("irregInputLevels", po::value<bool>(&irregInputLevels)->default_value(false), "irregular input levels")
        ("tVarLevels", po::value<bool>(&tVarLevels)->default_value(false), "variable input level sim time")
        ("dtVarLevels", po::value<bool>(&dtVarLevels)->default_value(false)," use different tstep for each input level")
        ("nInput", po::value<unsigned long>(&nInput)->default_value(1), " number of input sites")
        ("inputLevelsFn", po::value<string>(&inputLevelsFn),"filename of irregular input levels (need to set irregInputLevels, dtVarLevels or tVarLevels flag)")
        ("inputLinspace,i", po::value<vector<double>>()->multitoken()->composing(), "evenly spaced input level, start, total steps, end")
		("runTime,t", po::value<double>(), "simulation time for each input level")
        ("dt", po::value<double>(), "dt time step for each run")
        ("tVar", po::value<bool>(&tVar)->default_value(false), " use temporally variable input")
        ("pVar", po::value<bool>(&pVar)->default_value(false), " use positionally variable input")
        ("extendVar", po::value<bool>(&extendVar)->default_value(false), " extend pVar or tVar to other input levels")
        ("inputFn", po::value<string>(&inputFn),"filename of variable input table (need to set tVar or pVar flag)")
        ("reformatInputFn", po::value<string>(&reformatInputFn)->default_value("readoutDimension.bin"),"filename for storing trial lengths")
        ("nNeuron", po::value<unsigned long>(&nNeuron)->default_value(1),"number of neurons in simulation")
        ("withDend", po::value<bool>(&withDend)->default_value(false), "input direct on neuron's dendrites")
        ("conFn", po::value<string>(&conFn), "presynaptic connections binary data,repeat{pre;preID;(#dend;dendID);}")
        ("strFn", po::value<string>(&strFn), " connection strength binary data file, must match with conFn")
        ("seed", po::value<unsigned long>(&seed), " seed for the simulation");
    cmdLineOptions.add(configFileOptions);
    cmdLineOptions.add(initialGenerics);
}
int input_args::read(int argc, char **argv) {
    ifstream cfgFile;
    vector<vector<double>> vars;
    vector<unsigned long> column;
    vector<double> inputLinspace;
    po::store(po::parse_command_line(argc, argv, cmdLineOptions), vm);
    po::notify(vm);
    cfgFile.open(configFn);
    if (cfgFile) {
        store(po::parse_config_file(cfgFile, configFileOptions, true), vm);
        po::notify(vm);
    } else {
        cout << "cannot open configuration file: " << configFn << endl; 
        return 0;
    }
    if (seed<=0) {
	    seed = static_cast<unsigned int>(std::time(NULL));
    }
    // input levels
    if (irregInputLevels) {
        vars.push_back(inputLevel);
        if (vm.count("inputLinspace")) {
            cout << " using irregInputLevels flag, ignoring inputLinspace" << endl;
        }
    } else {
        if (!vm.count("inputLinspace")) {
            cout << "must provide a inputLinspace {begin,nlevels(,end)}" << endl;
            return 0;
        } else {
            inputLinspace = vm["inputLinspace"].as<vector<double>>();
            inputLevel.push_back(inputLinspace[0]);
            if (inputLinspace[1] > 1.5) {
                // inputLinspace[1] should be a "double" integer
                double dLevel = (inputLinspace[2] - inputLinspace[0])/(inputLinspace[1]-1);
                for (int i=1; i<inputLinspace[1]; i++) {
                    inputLevel.push_back(inputLinspace[0] + dLevel*i);
                }
                assert(abs(inputLevel.back()-inputLinspace[2]) < 1e-10);
            }
        }
    }
    if (tVarLevels) {
        vars.push_back(runTime);
        if (vm.count("runTime")) {
            cout << " using tVarLevels flag, ignoring runTime" << endl;
        }
    } else {
    }
    if (dtVarLevels) {
        vars.push_back(tstep);
        if (vm.count("dt")) {
            cout << " using dtVarLevels flag, ignoring single dt" << endl;
        }
    }
    if (irregInputLevels || tVarLevels || dtVarLevels) {
        if (!read_input_table(inputLevelsFn,column,vars)) {
            return 0;
        } else {
            cout << "# input levels: " << vars[0].size() << endl;
            if (irregInputLevels) {
                inputLevel = vars[0];
                if (tVarLevels) {
                    runTime = vars[1];
                    assert(runTime.size()==inputLevel.size());
                    if (dtVarLevels) {
                        tstep = vars[2];
                        assert(tstep.size()==inputLevel.size());
                    }
                }
            } else {
                if (tVarLevels) {
                    runTime = vars[0];
                    assert(runTime.size()==inputLevel.size());
                    if (dtVarLevels) {
                        tstep = vars[1];
                        assert(tstep.size()==inputLevel.size());
                    }
                } else {
                    if (dtVarLevels) {
                        tstep = vars[0];
                        assert(tstep.size()==inputLevel.size());
                    }
                }
            }
        }
        vars.clear();
    }
    if (!tVarLevels) {
        if (!vm.count("runTime")) {
            cout << " must provide a single runTime or an array of runTime" << endl;
            return 0;
        } else {
            runTime.assign(inputLevel.size(), vm["runTime"].as<double>());
            if (runTime[0] <= 0.0 ) {
                cout << " runTime must be positive" << endl;
                return 0;
            }
        }
    }
    if (!dtVarLevels) {
        if (!vm.count("dt")) {
            cout << "must provide a single dt (or an array of dt in levels.bin)" << endl;
            return 0;
        } else {
            tstep.assign(inputLevel.size(), vm["dt"].as<double>());
            if (tstep[0] <= 0.0) {
                cout << " dt must be positive" << endl;
                return 0;
            }
        }
    }
    // per input level
    if (tVar && pVar && (tVarLevels || dtVarLevels || extendVar)) {
        cout << " (tVar + pVar) is not compatible to tVarLevels, dtVarLevels, extendVar flag" << endl;
        cout << " run additional instances of simulation instead" << endl;
        return 0;
    }
    if (inputFn.empty()){
        cout << "no file for tVar available" << endl;
    } else {
        column.clear();
        // input dimension: site of input, time
        if (pVar && tVar && !tVarLevels && !dtVarLevels && !extendVar) {
            inputMode = "pt";
            column.push_back(round(runTime[0]/tstep[0]));
            if (!read_input_table(inputFn, column, input)) {
                return 0;
            }
            assert(input.size() == inputLevel.size());
        }
        // input dimension: trial, time
        if (!pVar && tVar && (tVarLevels || dtVarLevels)) {
            if (tVarLevels && !dtVarLevels) {
                inputMode = "t-T";
            }
            if (!tVarLevels && dtVarLevels) {
                inputMode = "t-dt";
            }
            if (tVarLevels && dtVarLevels) {
                inputMode = "t-Tdt";
            }
            for (int i=0; i<inputLevel.size(); i++) {
                column.push_back(round(runTime[i]/tstep[i]));
            }
            input.assign(inputLevel.size(),vector<double>());
            if (!read_input_table(inputFn, column, input)) {
                return 0;
            }
        }
        // input dimension: trial, site of input 
        if (pVar && !tVar && (tVarLevels || dtVarLevels)) {
            if (extendVar) {
                if (tVarLevels && !dtVarLevels) {
                    inputMode = "P-T";
                }
                if (!tVarLevels && dtVarLevels) {
                    inputMode = "P-dt";
                }
                input.assign(inputLevel.size(), vector<double>());
            } else {
                if (tVarLevels && !dtVarLevels) {
                    inputMode = "p-T";
                }
                if (!tVarLevels && dtVarLevels) {
                    inputMode = "p-dt";
                }
                if (tVarLevels && dtVarLevels) {
                    inputMode = "p-Tdt";
                }
                column.push_back(nInput);
            }
            if (!read_input_table(inputFn, column, input)) {
                return 0;
            }
            if (inputMode == "P-T" || inputMode == "P-dt") {
                for (int i=0; i<inputLevel.size(); i++) {
                    assert(input[i].size() == nInput);
                }
            }
            if (inputMode == "p-T" || inputMode == "p-dt") {
                assert(input.size() == 1);
                assert(input[0].size() == nInput);
            }
        }
        // input dimension (trial) time/site of input
        if ((!pVar && tVar || !tVar && pVar) && !tVarLevels && !dtVarLevels) {
            column.clear();
            if (!extendVar) {
                if (pVar) {
                    inputMode = "p";
                    column.push_back(nInput);
                }
                if (tVar) {
                    inputMode = "t";
                    column.push_back(round(runTime[0]/tstep[0]));
                }
            } else {
                if (pVar) {
                    inputMode = "P";
                    input.assign(inputLevel.size(),vector<double>());
                }
                if (tVar) {
                    inputMode = "T";
                    input.assign(inputLevel.size(),vector<double>());
                }
            }
            if (!read_input_table(inputFn, column, input)) {
                return 0;
            }
            if (inputMode == "t") {
                assert(input.size() == 1);
                for (int i=1; i<inputLevel.size(); i++) {
                    input.push_back(input[0]);
                }
            }
            if (inputMode == "T") {
                assert(input.size() == inputLevel.size());
                for (int i=0; i<input.size(); i++) {
                    assert(input[i].size() == input[0].size());
                }
                assert(input[0].size() == round(runTime[0]/tstep[0]));
                assert(input[0].size() == round(runTime.back()/tstep.back()));
            }
            if (inputMode == "p") {
                assert(input.size() == 1);
                for (int i=1; i<inputLevel.size(); i++) {
                    input.push_back(input[0]);
                }
            }
            if (inputMode == "P") {
                assert(input.size() == inputLevel.size());
                for (int i=0; i<input.size(); i++) {
                    assert(input[i].size() == nInput);
                }
            }
        }
    }
    if (vm.count("conFn")) {
        assert(nNeuron > 1);
        int skipping = 1;
        if (withDend) {
            skipping = 2;     
        }
        if (!vm.count("strFn")) {
            cout << "data of strength of connection also need to be provided." << endl;
            return 0;
        } else {
            column.clear();
            if (!read_input_table(conFn, column, preID)) {
                return 0;
            }
            column.clear();
            if (!read_input_table(strFn, column, preStr)) {
                return 0;
            }
            assert(preStr.size() == preID.size() - skipping*nNeuron);
        }
    }
    return 1;
}
int input_args::reformat_input_table(double tstep0) {
    ofstream outputfile;
    outputfile.open(reformatInputFn, ios::binary|ios::out);
    cout << "tstep" <<endl;
    for (int i=0; i<tstep.size(); i++) {
        cout << tstep[i] << endl;
    }
    assert(runTime.size() == inputLevel.size());
    assert(runTime.size() == tstep.size());
    int nTrial = runTime.size();
    vector<unsigned long> nDataPts;
    vector<unsigned long> nDataPtsSim;
    nDataPts.reserve(nTrial);
    for (int i=0; i<nTrial; i++) {
        nDataPts.push_back(static_cast<unsigned long>(round(runTime[i]/tstep0))+1);
        nDataPtsSim.push_back(static_cast<unsigned long>(round(runTime[i]/tstep[i]))+1);
    }
    if (outputfile.is_open()) {
        outputfile.write((char*)&nTrial, sizeof(int));
        outputfile.write((char*)&(nDataPtsSim[0]),nTrial*sizeof(unsigned long));
        outputfile.write((char*)&(nDataPts[0]),nTrial*sizeof(unsigned long));
        outputfile.write((char*)&(tstep[0]),nTrial*sizeof(double));
        outputfile.write((char*)&(inputLevel[0]),nTrial*sizeof(double));
        outputfile.write((char*)&(runTime[0]),nTrial*sizeof(double));
        outputfile.close();
        return 1;
    } else {
        cout << " failed to open " << reformatInputFn << " for writing readout dimensions " << endl;
        return 0;
    }
}
