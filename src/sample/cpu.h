#include <random>
#include <chrono>
#include <fstream>

#define timeNow() std::chrono::high_resolution_clock::now()

using namespace std::chrono;
double cpu_vE = 14.0f/3.0f; // dimensionaless (non-dimensionalized)
double cpu_vI = -2.0f/3.0f;
double cpu_vL = 0.0f, cpu_vT = 1.0f;

struct cConductanceShape {
    double riseTime[5], decayTime[5], deltaTau[5];
    cConductanceShape() {};
    cConductanceShape(double rt[], double dt[], unsigned int ng) {
        for (int i=0; i<ng; i++) {
            riseTime[i] = rt[i];
            decayTime[i] = dt[i];
            deltaTau[i] = dt[i] - rt[i];
        }
    }
    inline void compute_single_input_conductance(double *g, double *h, double f, double dt, unsigned int ig) {
        double etr = exp(-dt/riseTime[ig]);
        (*g) += f*decayTime[ig]*(exp(-dt/decayTime[ig])-etr)/deltaTau[ig];
        (*h) += f*etr;
    }
    inline void decay_conductance(double *g, double *h, double dt, unsigned int ig) {
        double etr = exp(-dt/riseTime[ig]);
        double etd = exp(-dt/decayTime[ig]);
        (*g) = (*g)*etd + (*h)*decayTime[ig]*(etd-etr)/deltaTau[ig];
        (*h) = (*h)*etr;
    }
};

std::uniform_real_distribution<double> distribution(0.0,1.0);
int h_set_input_time(double *inputTime, double dt, double rate, double &leftTimeRate, double &lastNegLogRand, std::minstd_rand &randGen) {
    int i = 0;
    double tau, dTau, negLogRand;
    tau = (lastNegLogRand - leftTimeRate)/rate;
    if (tau > dt) {
        leftTimeRate += (dt * rate);
        return i;
    } else do {
        inputTime[i] = tau;
        negLogRand = -log(distribution(randGen));
        dTau = negLogRand/rate;
        tau += dTau;
        i++;
        if (i == 10) {
            printf("exceeding max input per dt %i\n", 10);
            break;
        }
    } while (tau <= dt);
    lastNegLogRand = negLogRand;
    leftTimeRate = (dt - tau + dTau) * rate;
    return i;
}

void h_evolve_g(cConductanceShape &cond, double *g, double *h, double *f, double *inputTime, unsigned int nInput, double dt, unsigned int ig) {
    cond.decay_conductance(g, h, dt, ig); 
    for (int i=0; i<nInput; i++) {
        cond.compute_single_input_conductance(g, h, *f, dt-inputTime[i], ig);
    }
}

struct cpu_LIF {
    double v, v0;
    // type variable
    double tBack, tsp;
    double a1, b1;
    double a0, b0;
    cpu_LIF(double _v0): v(_v0) {
        tBack = -1.0f;
    }; 
    void runge_kutta_2(double dt);
    void set_p0(double _gE, double _gI, double _gL);
    void set_p1(double _gE, double _gI, double _gL);
    double eval0(double _v);
    double eval1(double _v);
    void reset_v();
    void compute_pseudo_v0(double dt);
    double compute_spike_time(double dt);
};

double cpu_step(cpu_LIF* lif, double dt, unsigned int id, double tRef) {
    lif->tsp = -1.0f;
    if (lif->tBack <= 0.0f) {
        // not in refractory period
        lif->runge_kutta_2(dt);
        if (lif->v > cpu_vT) {
            // crossed threshold
            lif->tsp = lif->compute_spike_time(dt); 
            lif->tBack = lif->tsp + tRef;
            //printf("neuron #%i fired initially\n", id);
        }
    }
    // return from refractory period
    if (lif->tBack > 0.0f && lif->tBack < dt) {
        lif->compute_pseudo_v0(dt);
        lif->runge_kutta_2(dt);
        lif->tBack = -1.0f;
    }
    // during refractory period
    if (lif->tBack > dt) {
        lif->reset_v(); 
        lif->tBack -= dt;
    } 
    return lif->tsp;
}

void cpu_LIF::runge_kutta_2(double dt) {
    double fk0 = eval0(v0);
    double fk1 = eval1(v0 + dt*fk0);
    v = v0 + dt*(fk0+fk1)/2.0f;
}

double cpu_LIF::compute_spike_time(double dt) {
    return (cpu_vT-v0)/(v-v0)*dt;
}

void cpu_LIF::compute_pseudo_v0(double dt) {
    v0 = (cpu_vL-tBack*(b0 + b1 - a1*b0*dt)/2.0f)/(1.0f+tBack*(-a0 - a1 + a1*a0*dt)/2.0f);
    runge_kutta_2(dt);
}

inline double cpu_get_a(double gE, double gI, double gL) {
    return gE + gI + gL;
}

inline double cpu_get_b(double gE, double gI, double gL) {
    return gE * cpu_vE + gI * cpu_vI + gL * cpu_vL;
}

void cpu_LIF::set_p0(double gE, double gI, double gL ) {
    a0 = cpu_get_a(gE, gI, gL);
    b0 = cpu_get_b(gE, gI, gL); 
}

void cpu_LIF::set_p1(double gE, double gI, double gL) {
    a1 = cpu_get_a(gE, gI, gL);
    b1 = cpu_get_b(gE, gI, gL); 
}

inline double cpu_eval_LIF(double a, double b, double v) {
    return -a * v + b;
}

double cpu_LIF::eval0(double _v) {
    return cpu_eval_LIF(a0,b0,_v);
}
double cpu_LIF::eval1(double _v) {
    return cpu_eval_LIF(a1,b1,_v);
}

void cpu_LIF::reset_v() {
    v = cpu_vL;
}

void cpu_version(int networkSize, double flatRate, unsigned int nstep, float dt, unsigned int h_nE, double s, unsigned long long seed, double ffsE, double ffsI) {
    s = 0.0f;
    unsigned int ngTypeE = 2;
    unsigned int ngTypeI = 1;
    double cpu_gL_E = 0.05f, cpu_gL_I = 0.1f; // kHz
    double gL, tRef;
    double cpu_tRef_E = 2.0f*dt, cpu_tRef_I = 1.0f*dt; // ms
    double *v = new double[networkSize];
    double *gE = new double[networkSize*ngTypeE];
    double *gI = new double[networkSize*ngTypeI];
    double *hE = new double[networkSize*ngTypeE];
    double *hI = new double[networkSize*ngTypeI];
    double *fE = new double[networkSize*ngTypeE];
    double *fI = new double[networkSize*ngTypeI];
    double *spikeTrain = new double[networkSize];
    std::ofstream v_file, spike_file, gE_file, gI_file;
    v_file.open("v_CPU.bin", std::ios::out|std::ios::binary);
    spike_file.open("s_CPU.bin", std::ios::out|std::ios::binary);
    gE_file.open("gE_CPU.bin", std::ios::out|std::ios::binary);
    gI_file.open("gI_CPU.bin", std::ios::out|std::ios::binary);
    std::minstd_rand **randGen = new std::minstd_rand*[networkSize];
    int *nInput = new int[networkSize];
    double *inputTime = new double[networkSize*10];
    double *logRand = new double[networkSize];
    double *lTR = new double[networkSize];

    double riseTimeE[2] = {1.0f, 5.0f}; // ms
    double riseTimeI[1] = {1.0f};
    double decayTimeE[2] = {3.0f, 80.0f};
    double decayTimeI[1] = {5.0f};
    cConductanceShape condE(riseTimeE, decayTimeE, ngTypeE);
    cConductanceShape condI(riseTimeI, decayTimeI, ngTypeI);

    printf("cpu version started\n");
    cpu_LIF **lif = new cpu_LIF*[networkSize];
    for (unsigned int i=0; i<networkSize; i++) {
        randGen[i] = new std::minstd_rand(seed+i);
        logRand[i] = -log(distribution(*randGen[i]));
        v[i] = 0.0f;
        lif[i] = new cpu_LIF(v[i]);
        gE[i] = 0.0f;
        gI[i] = 0.0f;
        hE[i] = 0.0f;
        hI[i] = 0.0f;
        fE[i] = ffsE;
        fI[i] = ffsI;
        lTR[i] = 0.0f;
    }
    //printf("cpu initialized\n");
    high_resolution_clock::time_point start = timeNow();
    v_file.write((char*)v, networkSize * sizeof(double));
    gE_file.write((char*)gE, networkSize * ngTypeE * sizeof(double));
    gI_file.write((char*)gI, networkSize * ngTypeI * sizeof(double));
    int inputEvents = 0;
    int outputEvents = 0;
    for (unsigned int istep=0; istep<nstep; istep++) {
        for (unsigned int i=0; i<networkSize; i++) {
            lif[i]->v0 = lif[i]->v;
            if (i<h_nE) {
                gL = cpu_gL_E;
                tRef = cpu_tRef_E;
            } else {
                gL = cpu_gL_I;
                tRef = cpu_tRef_I;
            }
            double gE_t, gI_t;
            if (istep == 0){
                lif[i]->set_p0(0, 0, gL);
            } else {
                gE_t = 0.0f;
                for (int ig = 0; ig < ngTypeE; ig++) {
                    int gid = networkSize*ig + i;
                    gE_t += gE[gid];
                }
                gI_t = 0.0f;
                for (int ig = 0; ig < ngTypeI; ig++) {
                    int gid = networkSize*ig + i;
                    gI_t += gI[gid];
                }
                lif[i]->a0 = cpu_get_a(gE_t, gI_t, gL);
                lif[i]->b0 = cpu_get_b(gE_t, gI_t, gL);
            }
            nInput[i] = h_set_input_time(&(inputTime[i*10]), dt, flatRate, lTR[i], logRand[i], *randGen[i]);
            inputEvents += nInput[i];
            /* evolve g to t+dt with ff input only */
            gE_t = 0.0f;
            #pragma roll
            for (int ig=0; ig<ngTypeE; ig++) {
                int gid = networkSize*ig + i;
                h_evolve_g(condE, &(gE[gid]), &(hE[gid]), &(fE[gid]), &(inputTime[i*10]), nInput[i], dt, ig);
                gE_t += gE[gid];
            }
            gI_t = 0.0f; 
            // no feed-forward inhibitory input (setting nInput = 0)
            #pragma roll
            for (int ig=0; ig<ngTypeI; ig++) {
                int gid = networkSize*ig + i;
                h_evolve_g(condI, &(gI[gid]), &(hI[gid]), &(fI[gid]), &(inputTime[i*10]), 0, dt, ig);
                gI_t += gI[gid];
            }
            //printf("i-%i, gI = %f\n", i, gI_t);
            lif[i]->set_p1(gE_t, gI_t, gL);
            spikeTrain[i] = cpu_step(lif[i], dt, i, tRef);
            v[i] = lif[i]->v;
        }
        for (unsigned int i=0; i<networkSize; i++) {
            double g_end, h_end;
            if (spikeTrain[i] > 0.0f) {
                outputEvents ++;
                if (i < h_nE) {
                    //printf("exc-%i fired\n", i);
                    #pragma unroll
                    for (int ig=0; ig<ngTypeE; ig++) {
                        g_end = 0;
                        h_end = 0;
                        condE.compute_single_input_conductance(&g_end, &h_end, s, dt-lif[i]->tsp, ig);
                        for (int ii = 0; ii < networkSize; ii++) {
                            int gid = networkSize*ig+ii;
                            gE[gid] += g_end;
                            hE[gid] += h_end;
                        }
                    }
                } else {
                    //printf("inh-%i fired\n", i);
                    #pragma unroll
                    for (int ig=0; ig<ngTypeI; ig++) {
                        g_end = 0;
                        h_end = 0;
                        condI.compute_single_input_conductance(&g_end, &h_end, s, dt-lif[i]->tsp, ig);
                        for (int ii = 0; ii < networkSize; ii++) {
                            int gid = networkSize*ig+ii;
                            gI[gid] += g_end;
                            hI[gid] += h_end;
                        }
                    }
                }
            }
        }
        v_file.write((char*)v, networkSize * sizeof(double));
        spike_file.write((char*)spikeTrain, networkSize * sizeof(int));
        gE_file.write((char*)gE, networkSize * ngTypeE * sizeof(double));
        gI_file.write((char*)gI, networkSize * ngTypeI * sizeof(double));
        printf("\r stepping %3.1f%%", 100.0f*float(istep+1)/nstep);
    }
    printf("\n");
    printf("input events rate %fkHz\n", float(inputEvents)/(dt*nstep*networkSize));
    printf("output events rate %fHz\n", float(outputEvents)*1000.0f/(dt*nstep*networkSize));
    auto cpuTime = duration_cast<microseconds>(timeNow()-start).count();
    printf("cpu version time cost: %3.1fms\n", static_cast<double>(cpuTime)/1000.0f);
    /* Cleanup */
    printf("Cleaning up\n");
    if (v_file.is_open()) v_file.close();
    if (spike_file.is_open()) spike_file.close();
    if (gE_file.is_open()) gE_file.close();
    if (gI_file.is_open()) gI_file.close();
    delete []v;
    delete []gE;
    delete []gI;
    delete []hE;
    delete []hI;
    delete []fE;
    delete []fI;
    delete []spikeTrain;
    for (unsigned int i=0; i<networkSize; i++) {
        delete []randGen[i];
    }
    delete []randGen;
    for (unsigned int i=0; i<networkSize; i++) {
        delete []lif[i];
    }
    delete []lif;
    delete []nInput;
    delete []inputTime;
    delete []logRand;
    delete []lTR;
    printf("Memories freed\n");
}
