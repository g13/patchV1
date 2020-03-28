#ifndef MODEL_CONST_H
#define MODEL_CONST_H

#define vE 14.0f/3.0f
#define vI -2.0f/3.0f
#define vT 1.0f
#define vL 0.0f
#define gL_E 0.05f
#define gL_I 0.08f
#define tRef_E 2.0f
#define tRef_I 1.0f

#define max_nLearnTypeFF 2 // maximum number of feedforward excitatory learning types
#define max_nLearnTypeE 2 // cortical excitatory
#define max_nLearnTypeI 2 // inhibitory
#define max_nLearnType 2 // max 
#define sum_nType (max_nTypeE+max_nTypeI) // sum

#define max_ngTypeFF 2 // excitatory feedforward conductance types
#define max_ngTypeE 2 // excitatory coritcal
#define max_ngTypeI 2 // inhibitory
#define max_ngType 2 // max
//#define max_ngType (max_ngTypeFF+max_ngTypeE+max_ngTypeI) // sum 
#define EPS 1e-14

#endif
