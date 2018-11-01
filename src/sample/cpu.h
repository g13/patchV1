#include <random>
#include <ctime>

double cpu_vE = 14.0f/3.0f; // dimensionaless (non-dimensionalized)
double cpu_vI = -2.0f/3.0f;
double cpu_vL = 0.0f, cpu_vT = 1.0f;

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
        if (i == MAX_FFINPUT_PER_DT) {
            printf("exceeding max input per dt %i\n", MAX_FFINPUT_PER_DT);
            break;
        }
    } while (tau <= dt);
    lastNegLogRand = negLogRand;
    leftTimeRate = (dt - tau + dTau) * rate;
    return i;
}

void h_evolve_g(ConductanceShape &cond, double *g, double *h, double *f, double *inputTime,unsigned int nInput, double dt, unsigned int ig) {
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
    cpu_LIF(double _v0): v0(_v0) {
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
    } else {
        // during refractory period
        if (lif->tBack > dt) {
            lif->reset_v(); 
        } 
    }
    // return from refractory period
    if (lif->tBack > 0.0f && lif->tBack < dt) {
        lif->compute_pseudo_v0(dt);
        lif->runge_kutta_2(dt);
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

//inline double eval_LIF(double a, double b, double v) {
//    return -a * v + b;
//}

double cpu_LIF::eval0(double _v) {
    return eval_LIF(a0,b0,_v);
}
double cpu_LIF::eval1(double _v) {
    return eval_LIF(a1,b1,_v);
}

void cpu_LIF::reset_v() {
    v = cpu_vL;
}

void cpu_version(int networkSize, ConductanceShape condE, ConductanceShape condI, double flatRate, unsigned int nstep, float dt, unsigned int ngTypeE, unsigned int ngTypeI, unsigned int nE, double s) {
    double gL_E = 0.05f, gL_I = 0.1f; // kHz
    double gL, tRef;
    double tRef_E = 2.0f, tRef_I = 1.0f; // ms
    double *v = new double[networkSize];
    double *gE = new double[networkSize*ngTypeE];
    double *gI = new double[networkSize*ngTypeI];
    double *hE = new double[networkSize*ngTypeE];
    double *hI = new double[networkSize*ngTypeI];
    double *fE = new double[networkSize*ngTypeE];
    double *fI = new double[networkSize*ngTypeI];
    double *a = new double[networkSize];
    double *b = new double[networkSize];
    double *spikeTrain = new double[networkSize];
    std::ofstream v_file, spike_file, gE_file, gI_file;
    v_file.open("v_CPU.bin", std::ios::out|std::ios::binary);
    spike_file.open("s_CPU.bin", std::ios::out|std::ios::binary);
    gE_file.open("gE_CPU.bin", std::ios::out|std::ios::binary);
    gI_file.open("gI_CPU.bin", std::ios::out|std::ios::binary);
    std::minstd_rand **randGen = new std::minstd_rand*[networkSize];
    int *nInput = new int[networkSize];
    double *inputTime = new double[networkSize*MAX_FFINPUT_PER_DT];
    double *logRand = new double[networkSize];
    double *lTR = new double[networkSize];
    unsigned long long seed = 183765712;
    for (unsigned int i=0; i<networkSize; i++) {
        randGen[i] = new std::minstd_rand(seed+i);
        logRand[i] = distribution(*randGen[i]);
        gE[i] = 0.0f;
        gI[i] = 0.0f;
        hE[i] = 0.0f;
        hI[i] = 0.0f;
        fE[i] = 1.0f;
        fI[i] = 1.0f;
    }
    std::clock_t start;
    start = std::clock();
    for (unsigned int istep=0; istep<nstep; istep++) {
        for (unsigned int i=0; i<networkSize; i++) {
            cpu_LIF lif(v[i]);
            if (i<nE) {
                gL = gL_E;
                tRef = tRef_E;
            } else {
                gL = gL_I;
                tRef = tRef_I;
            }
            if (istep == 0){
                lif.set_p0(0, 0, gL);
            } else {
                lif.a0 = a[i];
                lif.b0 = b[i]; 
            }
            nInput[i] = h_set_input_time(&(inputTime[i*MAX_FFINPUT_PER_DT]), dt, flatRate, lTR[i], logRand[i], *randGen[i]);
            double gE_t = 0.0f;
            #pragma roll
            for (int ig=0; ig<ngTypeE; ig++) {
                int gid = networkSize*ig + i;
                h_evolve_g(condE, &(gE[gid]), &(hE[gid]), &(fE[gid]), &(inputTime[i*MAX_FFINPUT_PER_DT]), nInput[i], dt, ig);
                gE_t += gE[gid];
            }
            double gI_t = 0.0f; 
            #pragma roll
            for (int ig=0; ig<ngTypeI; ig++) {
                int gid = networkSize*ig + i;
                h_evolve_g(condI, &(gI[gid]), &(hI[gid]), &(fI[gid]), &(inputTime[i*MAX_FFINPUT_PER_DT]), nInput[i], dt, ig);
                gI_t += gI[gid];
            }
            lif.set_p1(gE_t, gI_t, gL);
            spikeTrain[i] = cpu_step(&lif, dt, i, tRef);
            v[i] = lif.v;

            double g_end, h_end;
            if (spikeTrain[i] > 0.0f) {
                if (i < nE) {
                    #pragma unroll
                    for (int ig=0; ig<ngTypeE; ig++) {
                        condE.compute_single_input_conductance(&g_end, &h_end, s, lif.tsp, ig);
                        for (int ii = 0; ii < networkSize; ii++) {
                            int gid = networkSize*ig+ii;
                            gE[gid] += g_end;
                            hE[gid] += h_end;
                        }
                    }
                } else {
                    #pragma unroll
                    for (int ig=0; ig<ngTypeI; ig++) {
                        condI.compute_single_input_conductance(&g_end, &h_end, s, lif.tsp, ig);
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
    }
    printf("cpu Time: %fms\n", (std::clock() - start)/(double)(CLOCKS_PER_SEC/1000)/(dt*nstep));
    delete []gE;
    delete []gI;
    delete []v;
    delete []spikeTrain;
    for (unsigned int i=0; i<networkSize; i++) {
        delete []randGen[i];
    }
    delete []*randGen;
    delete []logRand;
    delete []nInput;
    delete []a;
    delete []b;
}
