#include "input_args_CA1.h"

input_args_CA1::input_args_CA1() {
    po::options_description CA1_options;
    CA1_options.add_options()
		("libFile,l",po::value<string>(&libFile),"neuron library")
        ("vInit,v", po::value<unsigned int>(&vInit), "index of intial voltage in vRange")
        ("vThres", po::value<double>(&vThres), "spiking threshold")
        ("vRest", po::value<double>(&vRest), "resting threshold")
        ("tRef",po::value<double>(&tRef),"refractory period")
        ("trans",po::value<double>(&trans),"transient VClamp time")
        ("trans0",po::value<double>(&trans0),"default transient VClamp time")
        ("rLinear",po::value<double>(&rLinear),"linear model spiking threshold")
        ("rBiLinear", po::value<double>(&rBiLinear), "bilinear model spiking threshold")
        ("vTol",po::value<double>(&vTol),"crossing threshold tolerance")
        ("vBuffer",po::value<double>(&vBuffer),"return threshold buffer")
        ("dendClampRatio",po::value<double>(&dendClampRatio),"dendrite dV contribution ratio")
        ("afterSpikeBehavior",po::value<int>(&afterSpikeBehavior)->default_value(0), "0:no linear or bilinear extension after spike. 1:no bilinear extension. 2: all extend.")
        ("kVStyle",po::value<int>(&kVStyle)->default_value(0), "bilinear0 after spike kV style, 0: kV0 style. 1: kV style.")
		("ignoreT", po::value<double>(&ignoreT),"ingore time while applying bilinear rules");
        
    cmdLineOptions.add(CA1_options);
    configFileOptions.add(CA1_options);
}
