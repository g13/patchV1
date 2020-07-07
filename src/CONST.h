#ifndef CONST_H
#define CONST_H

#define vE 14.0f/3.0f
#define vI -2.0f/3.0f
#define vL 0.0f

#define MAX_NTYPE 2

#define MAX_NLEARNTYPE_FF_I 1 // ff to inhibitory only
#define MAX_NLEARNTYPE_FF_E 1 // ff to excitatory only
#define MAX_NLEARNTYPE_FF 1 // maximum of either E or I ff learning types
#define SUM_NLEARNTYPE_FF (MAX_NLEARNTYPE_FF_I + MAX_NLEARNTYPE_FF_E)

#define MAX_NLEARNTYPE_E 2 // cortical excitatory to excitatory
#define MAX_NLEARNTYPE_Q 1 // cortical inhibitory to excitatory 

//#define MAX_NLEARNTYPE 2 // max 
//#define SUM_NTYPE (MAX_NTYPEE+MAX_NTYPEI) // sum

#define MAX_NGTYPE_FF 2 // excitatory feedforward conductance types
#define MAX_NGTYPE_E 2 // excitatory coritcal
#define MAX_NGTYPE_I 1 // inhibitory
#define MAX_NGTYPE 2 // max
#define EPS 1e-14
#define SQRT2 1.4142135623730951

const int max_nType = MAX_NTYPE;

const int max_nLearnTypeFF_I = MAX_NLEARNTYPE_FF_I;
const int max_nLearnTypeFF_E = MAX_NLEARNTYPE_FF_E;
const int max_nLearnTypeFF = MAX_NLEARNTYPE_FF;
const int sum_nLearnTypeFF = SUM_NLEARNTYPE_FF;

const int max_nLearnTypeE = MAX_NLEARNTYPE_E;
const int max_nLearnTypeQ = MAX_NLEARNTYPE_Q;

//const int max_nLearnType = MAX_NLEARNTYPE;

const int max_ngTypeFF = MAX_NGTYPE_FF;
const int max_ngTypeE = MAX_NGTYPE_E;
const int max_ngTypeI = MAX_NGTYPE_I;
const int max_ngType = MAX_NGTYPE;
const Float eps = EPS;
const Float sqrt2 = SQRT2;

#endif
