#include "cpu.h"
/* === RAND === Uncomment all RANDs if curand can ensure cpu and gpu generate the same rands
    int set_input_time(double *inputTime, double dt, double rate, double &leftTimeRate, double &lastNegLogRand, curandGenerator_t &randGen) {
        int i = 0;
        double tau, dTau, negLogRand;
        tau = (lastNegLogRand - leftTimeRate)/rate;
        if (tau > dt) {
            leftTimeRate += (dt * rate);
            return i;
        } else do {
            inputTime[i] = tau;
            curandGenerateUniformDouble(randGen, &negLogRand, 1);
            negLogRand = -log(negLogRand);        
            dTau = negLogRand/rate;
            tau += dTau;
            i++;
            if (i == MAX_FFINPUT_PER_DT) {
                printf("exceeding max input per dt %i\n", 10);
                break;
            }
        } while (tau <= dt);
        lastNegLogRand = negLogRand;
        leftTimeRate = (dt - tau + dTau) * rate;
        return i;
    }
*/

void evolve_g(cConductanceShape &cond, double *g, double *h, double *f, double *inputTime, int nInput, double dt, double dt0, unsigned int ig) {
    // dt0: with respect to the time frame of inputTime (0)
    // dt: with respect to time frame of actual integration 
    cond.decay_conductance(g, h, dt, ig);  
    for (int i=0; i<nInput; i++) {
        cond.compute_single_input_conductance(g, h, *f, dt0-inputTime[i], ig);
    }
}

double cpu_step(cpu_LIF* lif, double dt, double tRef, unsigned int id, double gE, double gI) {
    lif->tsp = dt;
    // not in refractory period
    if (lif->tBack < dt) {
        // return from refractory period
        if (lif->tBack > 0.0f) {
            lif->compute_pseudo_v0(dt, 0);
            lif->tBack = -1.0f;
        }
        lif->runge_kutta_2(dt);
        while (lif->v > vT && lif->tBack < 0.0f) {
            // crossed threshold
            if (lif->v > vE) {
				printf("#%i something is off gE = %f, gI = %f, v = %f\n", id, gE, gI, lif->v);
                lif->v = vE;
            }

            lif->compute_spike_time(dt, 0); 
            lif->tBack = lif->tsp + tRef;
            //printf("neuron #%i fired initially\n", id);
            //assert(lif->tBack > 0);
            if (lif->tBack < dt) {
                // refractory period ended during dt
                lif->compute_v(dt, 0);
                lif->tBack = -1.0f;
                if (lif->v > vT) {
                    printf("multiple spike in one time step, only the last spike is counted, refractory period = %f ms, dt = %f\n", tRef, dt);
                    //assert(lif->v <= vT);
                }
            }
        }
    } 
    if (lif->tBack >= dt) {
        // during refractory period
        lif->reset_v(); 
        lif->tBack -= dt;
    } 
    return lif->tsp;
}

double cpu_dab(cpu_LIF* lif, double pdt0, double dt, double pdt, double dt0, double tRef, unsigned int id, double gE, double gI) {
    // dt0: with respect to the original start of time step (0) 
    // dt: with respect to time frame of actual integration 
    // time --->
    // |      |              |           |
    // 0    pdt0 <---dt---->pdt        dt0
    // tsp is set to be relative to t=0 (of the current step instead of the beginning of the simulation)
    lif->tsp = dt0;
    lif->correctMe = true;
    lif->spikeCount = 0;
    // not in refractory period
    if (lif->tBack < pdt) {
        // return from refractory period
        if (lif->tBack > 0.0f) {
            assert(lif->tBack>pdt0);
            lif->compute_pseudo_v0(dt, pdt0);
            lif->tBack = -1.0f;
        }
        lif->runge_kutta_2(dt);
        if (lif->v > vT) {
            // crossed threshold
            if (lif->v > vE) {
				//printf("#%i shoots above vE, something is off gE = %f, gI = %f, v = %f\n", id, gE, gI, lif->v);
                lif->v = vE;
            }

            lif->compute_spike_time(dt, pdt0); 
            // dabbing not commiting, doest not reset v or recored tBack, TBD by spike correction.
        } else {
            if (lif->v < vI) {
			    //printf("#%i shoots below vI, something is off gE = %f, gI = %f, v = %f\n", id, gE, gI, lif->v);
                lif->v = vI;
            }
        }
    } else {
        lif->reset_v(); 
        if (lif->tBack >= dt0) {
            // during refractory period
            lif->tBack -= dt0;
            lif->correctMe = false;
        }
    } 
    return lif->tsp;
}

double eval_LIF(double a, double b, double v) {
	return -a * v + b;
}

double get_a(double gE, double gI, double gL) {
	return gE + gI + gL;
}

double get_b(double gE, double gI, double gL) {
	return gE * vE + gI * vI + gL * vL;
}

double compute_v1(double dt, double a0, double b0, double a1, double b1, double v, double t) {
	double A = 1.0 + (a0*a1*dt - a0 - a1) * dt / 2.0f;
	double B = (b0 + b1 - a1 * b0*dt) * dt / 2.0f;
	return (B*(t - dt) - A * v*dt) / (t - dt - A * t);
}


void cpu_LIF::runge_kutta_2(double dt) {
    double fk0 = eval0(v0);
    v_hlf = v0 + dt*fk0;
    double fk1 = eval1(v_hlf);
    v = v0 + dt*(fk0+fk1)/2.0f;
}

void cpu_LIF::compute_spike_time(double dt, double pdt0) {
    tsp = pdt0 + (vT-v0)/(v-v0)*dt;
}

void cpu_LIF::compute_v(double dt, double pdt0) {
    v = compute_v1(dt, a0, b0, a1, b1, vL, tBack-pdt0);
}

void cpu_LIF::compute_pseudo_v0(double dt, double pdt0) {
    v0 = (vL-(tBack-pdt0)*(b0 + b1 - a1*b0*dt)/2.0f)/(1.0f+(tBack-pdt0)*(-a0 - a1 + a1*a0*dt)/2.0f);
}

void cpu_LIF::trans_p1_to_p0() {
    a0 = a1;
    b0 = b1;
}

void cpu_LIF::set_p0(double gE, double gI, double gL ) {
    a0 = get_a(gE, gI, gL);
    b0 = get_b(gE, gI, gL); 
}

void cpu_LIF::set_p1(double gE, double gI, double gL) {
    a1 = get_a(gE, gI, gL);
    b1 = get_b(gE, gI, gL); 
}

double cpu_LIF::eval0(double _v) {
    return eval_LIF(a0,b0,_v);
}
double cpu_LIF::eval1(double _v) {
    return eval_LIF(a1,b1,_v);
}

void cpu_LIF::reset_v() {
    v = vL;
}

unsigned int find1stAfter(double spikeTrain[], unsigned int n, double min) {
    // in case no spike i=0
    unsigned int i = 0;
    for (int j=0; j<n; j++) {
        if (min - spikeTrain[j] > EPS) {
            min = spikeTrain[j];
            i = j;
        }
    }
    return i;
}

// Spike-spike correction
unsigned int cpu_ssc(cpu_LIF* lif[], double v[], double gE0[], double gI0[], double hE0[], double hI0[], double gE1[], double gI1[], double hE1[], double hI1[], double fE[], double fI[], double preMat[], unsigned int networkSize, cConductanceShape condE, cConductanceShape condI, int ngTypeE, int ngTypeI, double inputTime[], int nInput[], double spikeTrain[], unsigned int nE, double dt) {
    double *v0 = new double[networkSize];
    double *wSpikeTrain = new double[networkSize];
    unsigned int *idTrain = new unsigned int[networkSize];
    unsigned int corrected_n = 0;
    unsigned int n = 0;
    for (unsigned int i=0; i<networkSize; i++) {
        if (dt - lif[i]->tsp > EPS) {
            wSpikeTrain[n] = lif[i]->tsp;
            idTrain[n] = i;
            n++;
        }
        // collect v0 in case of reclaim
        v0[i] = lif[i]->v0;
    }
    double *g_end, *h_end;
    if (ngTypeE > ngTypeI) {
        g_end = new double[ngTypeE];
        h_end = new double[ngTypeE];
    } else {
        g_end = new double[ngTypeI];
        h_end = new double[ngTypeI];
    }
    double pdt0 = 0.0f; // old pdt
    int* iInput = new int[networkSize];
    int* lastInput = new int[networkSize]();
    double* gE = new double[networkSize*ngTypeE];
    double* hE = new double[networkSize*ngTypeE];
    double* gI = new double[networkSize*ngTypeI];
    double* hI = new double[networkSize*ngTypeI];
    int r = 0;
    while (n>0) {
        if (r>2*networkSize) {
            printf("fucked\n");
        }
        // get the first spike, candidate to output spikeTrain
        unsigned int head = find1stAfter(wSpikeTrain, n, dt);
        // candidate index = i
        unsigned int i = idTrain[head];
        double pdt = wSpikeTrain[head]; // relative to [0, dt] 
        double dpdt = pdt - pdt0;
        printf("pdt - pdt0  = %e, %i, %f, %f, %i, %i, %i\n", pdt - pdt0, i, gE0[i], gI0[i], corrected_n, r, n);
        // |      |             |             |
        // 0     pdt0<--dpdt-->pdt<---ddt---->dt
        // candidate will be cancelled if another spike reclaimed before the candidate
        bool reclaimed = false;
        // restep to fill-up the wSpikeTrain
        n = 0;
        for (int j=0; j<networkSize; j++) {
            if (lif[j]->correctMe) {
                double gL, tRef;
                if (j<nE) {
                    gL = gL_E;
                    tRef = tRef_E;
                } else {
                    gL = gL_I;
                    tRef = tRef_I;
                }
                if (i!=j) {
                // excluding neurons in refractory period, nothing new here except in advancing the conductance putatively and set_p0 that wont be used by the candidate, neuron i 
                    double gE_t = 0.0f;
                    #pragma unroll
                    for (int ig=0; ig<ngTypeE; ig++) {
                        unsigned int gid = networkSize*ig + j;
                        gE_t = gE0[gid];
                    }
                    double gI_t = 0.0f;
                    #pragma unroll
                    for (int ig=0; ig<ngTypeI; ig++) {
                        unsigned int gid = networkSize*ig + j;
                        gI_t = gI0[gid];
                    }
                    lif[j]->set_p0(gE_t, gI_t, gL);
                }
                // evolve g h and lastInput to pdt (if no spike is reclaimed then update g0 and h0 and lastInput)
                // determine ext input range [pdt0, pdt]
                iInput[j] = lastInput[j];
                if  (iInput[j] < nInput[j]) {
                    while (inputTime[j*MAX_FFINPUT_PER_DT+iInput[j]] < pdt) {
                        iInput[j]++;
                        if (iInput[j] < nInput[j]) break;
                    }
                }
                for (int ig=0; ig<ngTypeE; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    gE[gid] = gE0[gid];
                    hE[gid] = hE0[gid];
                    evolve_g(condE, &(gE[gid]), &(hE[gid]), &(fE[gid]), &(inputTime[j*MAX_FFINPUT_PER_DT+lastInput[j]]), iInput[j]-lastInput[j], dpdt, pdt, ig);
                }
                // no feed-forward inhibitory input (setting nInput = 0)
                for (int ig=0; ig<ngTypeI; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    gI[gid] = gI0[gid];
                    hI[gid] = hI0[gid];
                    evolve_g(condI, &(gI[gid]), &(hI[gid]), &(fI[gid]), inputTime, 0, dpdt, pdt, ig);
                }
                double gE_t = 0.0f;
                #pragma unroll
                for (int ig=0; ig<ngTypeE; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    gE_t += gE[gid];
                }
                double gI_t = 0.0f; 
                #pragma unroll
                for (int ig=0; ig<ngTypeI; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    gI_t += gI[gid];
                }
                lif[j]->set_p1(gE_t, gI_t, gL);
                // p1 is going to be used by candidate i if its tBack is still smaller than dt
                if (i!=j) {
                    // exclude the candidate itself for updating voltage or tsp
                    lif[j]->v0 = v0[j];
                    cpu_dab(lif[j], pdt0, dpdt, pdt, dt, tRef, j, gE_t, gI_t); 
                    if (dt - lif[j]->tsp > EPS ) {
                        reclaimed = true;
                        // ignore round off error
                        idTrain[n] = j;
                        wSpikeTrain[n] = lif[j]->tsp;
                        n++;
                    } 
                }
            }
        }
        if (!reclaimed) {
            // if ndeed no spike comes before neuron i with the smaller pdt interpolation, commit into spikeTrain
            corrected_n++;
            spikeTrain[i] = wSpikeTrain[head];
            // realign tsp to be relative to the start of the time step, 0
            lif[i]->tsp = wSpikeTrain[head];
            cConductanceShape *cond;
            lif[i]->spikeCount++;
            if (lif[i]->spikeCount > 1) {
                printf("multiple spikes in one dt, consider smaller time step than %fms\n", dt);
            }
            if (i < nE) {
                lif[i]->tBack = lif[i]->tsp + tRef_E;
            } else {
                lif[i]->tBack = lif[i]->tsp + tRef_I;
            }
            if (lif[i]->tBack > dt) {
                lif[i]->reset_v();
                lif[i]->tBack -= dt;
                lif[i]->correctMe = false;
                v[i] = lif[i]->v;
            }
		    //printf(", v1 new = %f\n", lif[i]->v);
            // set pdt0 for next
            pdt0 = pdt;
            // move on to next pdt period
            double ddt = dt - pdt;
            // prepare the conductance changes caused by the confirmed spike from neuron i
            int ngType;
            double *g, *h;
            if (i < nE) {
                cond = &condE;
                ngType = ngTypeE;
                g = gE1;
		    	h = hE1;
            } else {
                cond = &condI;
                ngType = ngTypeI;
                g = gI1;
		    	h = hI1;
            }
            for (int ig = 0; ig<ngType; ig++) {
                g_end[ig] = 0.0f;
                h_end[ig] = 0.0f;
                cond->compute_single_input_conductance(&(g_end[ig]), &(h_end[ig]), 1.0f, ddt, ig);
            }
            // update v, tsp and conductance for the time interval of [pdt, dt]
            n = 0;
            for (int j=0; j<networkSize; j++) {
                // commit g0 h0
                for (int ig=0; ig<ngTypeE; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    gE0[gid] += gE[ig];
                    hE0[gid] += hE[ig];
                }
                for (int ig=0; ig<ngTypeI; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    gI0[gid] += gI[ig];
                    hI0[gid] += hI[ig];
                }
                // commit lastInput
                lastInput[j] = iInput[j];
                // new cortical input
                for (int ig=0; ig<ngType; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    g[gid] += g_end[ig] * preMat[i*networkSize + j];
                    h[gid] += h_end[ig] * preMat[i*networkSize + j];
                }
                if (lif[j]->correctMe) {
                    double gL, tRef;
                    if (j<nE) {
                        gL = gL_E;
                        tRef = tRef_E;
                    } else {
                        gL = gL_I;
                        tRef = tRef_I;
                    }
                    // if not in refractory period
                    lif[j]->v0 = lif[j]->v; // v0 is irrelevant if the neuron is coming back from spike, it will be reset in cpu_dab
                    lif[j]->trans_p1_to_p0();

                    double gE_t = 0.0f;
                    #pragma unroll
                    for (int ig = 0; ig < ngTypeE; ig++) {
                        int gid = networkSize*ig + j;
                        gE_t += gE1[gid];
                    }
                    double gI_t = 0.0f;
                    #pragma unroll
                    for (int ig = 0; ig < ngTypeI; ig++) {
                        int gid = networkSize*ig + j;
                        gI_t += gI1[gid];
                    }
                    lif[j]->set_p1(gE_t, gI_t, gL);
                    // dab
                    cpu_dab(lif[j], pdt, ddt, dt, dt, tRef, j, gE_t, gI_t);
                    // put back to [0,dt]
                    if (dt - lif[j]->tsp > EPS) { // equiv lif->v > vT
                        // ignore round off error
                        idTrain[n] = j;
                        wSpikeTrain[n] = lif[j]->tsp;
                        n++;
                    } else {
                        //update volts
                        v[j] = lif[j]->v;
                    }
                    // commits v0 for next advancement in 
                    v0[j] = lif[j]->v0;
                }
            }
        }
        r++;
    }

    for (unsigned int i=0; i<networkSize; i++) {
        if (v[i] < -0.1f) {
            printf("%i here to destroy, tBack = %f, tsp = %f, v = %f\n", i, lif[i]->tBack, lif[i]->tsp, v[i]);
            assert(v[i]>=0.0f);
        }
    }
    delete []g_end;
    delete []h_end;
    delete []iInput;
    delete []lastInput;
    delete []gE;
    delete []gI;
    delete []hE;
    delete []hI;
    delete []v0;
    delete []wSpikeTrain;
    delete []idTrain;
    return corrected_n;
}

void cpu_version(int networkSize, /* === RAND === flatRate */double dInput, unsigned int nstep, double dt, unsigned int nE, double preMat0[], double vinit[], double firstInput[], /* === RAND === unsigned long long seed, */ double ffsE, double ffsI, std::string theme, double inputRate) {
    unsigned int ngTypeE = 2;
    unsigned int ngTypeI = 1;
    double gL, tRef;
    double *v = new double[networkSize];
    double *gE0 = new double[networkSize*ngTypeE];
    double *gI0 = new double[networkSize*ngTypeI];
    double *gE1 = new double[networkSize*ngTypeE];
    double *gI1 = new double[networkSize*ngTypeI];
    double *gE_current, *gI_current, *gE_old, *gI_old;
    double *hE0 = new double[networkSize*ngTypeE];
    double *hI0 = new double[networkSize*ngTypeI];
    double *hE1 = new double[networkSize*ngTypeE];
    double *hI1 = new double[networkSize*ngTypeI];
    double *hE_current, *hI_current, *hE_old, *hI_old;
    double *fE = new double[networkSize*ngTypeE];
    double *fI = new double[networkSize*ngTypeI];
    double *dv = new double[networkSize];
    double *v_hlf = new double[networkSize];
    double *a0 = new double[networkSize];
    double *b0 = new double[networkSize];
    double *a1 = new double[networkSize];
    double *b1 = new double[networkSize];
    double *spikeTrain = new double[networkSize];
    double *preMat = new double[networkSize*networkSize];
    std::ofstream p_file, v_file, spike_file, gE_file, gI_file;
    p_file.open("p_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    v_file.open("v_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    spike_file.open("s_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    gE_file.open("gE_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    gI_file.open("gI_CPU" + theme + ".bin", std::ios::out|std::ios::binary);

    unsigned int nI = networkSize - nE;
    p_file.write((char*)&nE, sizeof(unsigned int));
    p_file.write((char*)&nI, sizeof(unsigned int));
    p_file.write((char*)&ngTypeE, sizeof(unsigned int));
    p_file.write((char*)&ngTypeI, sizeof(unsigned int));
    double dtmp = vL;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = vT;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = vE;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = vI;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = gL_E;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = gL_I;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = tRef_E;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = tRef_I;
    p_file.write((char*)&dtmp, sizeof(double));
    p_file.write((char*)&nstep, sizeof(unsigned int));
    p_file.write((char*)&dt, sizeof(double));
    p_file.write((char*)&inputRate, sizeof(double));

    double *inputTime = new double[networkSize*MAX_FFINPUT_PER_DT];
    /* === RAND === 
        curandGenerator_t *randGen = new curandGenerator_t[networkSize];
        int *nInput = new int[networkSize];
    */
	/* not used if not RAND */
    double *logRand = new double[networkSize];
    double *lTR = new double[networkSize];
    double riseTimeE[2] = {1.0f, 5.0f}; // ms
    double riseTimeI[1] = {1.0f};
    double decayTimeE[2] = {3.0f, 80.0f};
    double decayTimeI[1] = {5.0f};
    cConductanceShape condE(riseTimeE, decayTimeE, ngTypeE);
    cConductanceShape condI(riseTimeI, decayTimeI, ngTypeI);

    printf("cpu version started\n");
	/* === RAND ===
		high_resolution_clock::time_point rStart = timeNow();
		for (unsigned int i = 0; i < networkSize; i++) {
			curandCreateGeneratorHost(&randGen[i], CURAND_RNG_PSEUDO_MRG32K3A);
			curandSetPseudoRandomGeneratorSeed(randGen[i], seed + i);
			curandGenerateUniformDouble(randGen[i], &(logRand[i]), 1);
			logRand[i] = -log(logRand[i]);
			lTR[i] = 0.0f;
			printf("\r initializing curand Host generators %3.1f%%", 100.0f*float(i + 1) / networkSize);
		}
		double rTime = static_cast<double>(duration_cast<microseconds>(timeNow() - rStart).count());
		printf(" cost: %fms\n", rTime/1000.0f);
	*/
    cpu_LIF **lif = new cpu_LIF*[networkSize];
	high_resolution_clock::time_point iStart = timeNow();
    for (unsigned int i=0; i<networkSize; i++) {
        v[i] = vinit[i];
        lif[i] = new cpu_LIF(v[i]);
        lTR[i] = firstInput[i];
        for (int j=0; j<networkSize; j++) {
            preMat[i*networkSize+j] = preMat0[i*networkSize+j];
        }
    }
	double iTime = static_cast<double>(duration_cast<microseconds>(timeNow() - iStart).count());
	printf("initialization cost: %fms\n", iTime/1000.f);

    for (int ig = 0; ig < ngTypeE; ig++) {
        for (unsigned int i = 0; i < networkSize; i++) {
            unsigned int gid = ig*networkSize + i;
            gE0[gid] = 0.0f;
            hE0[gid] = 0.0f;
            fE[gid] = ffsE;
        }
    }
    for (int ig = 0; ig < ngTypeI; ig++) {
        for (unsigned int i = 0; i < networkSize; i++) {
            unsigned int gid = ig*networkSize + i;
            gI0[gid] = 0.0f;
            hI0[gid] = 0.0f;
            fI[gid] = ffsI;
        }
    }
    //printf("cpu initialized\n");
    high_resolution_clock::time_point start = timeNow();
    v_file.write((char*)v, networkSize * sizeof(double));
    gE_file.write((char*)gE0, networkSize * ngTypeE * sizeof(double));
    gI_file.write((char*)gI0, networkSize * ngTypeI * sizeof(double));
	int inputEvents = 0;
    int outputEvents = 0;
    double vTime = 0.0f;
	double wTime = 0.0f;
    double sTime = 0.0f;
    int *nInput = new int[networkSize];
	for (unsigned int istep = 0; istep < nstep; istep++) {
		if (istep % 2 == 0) {
			gE_current = gE1;
			hE_current = hE1;
			gE_old = gE0;
			hE_old = hE0;
			gI_current = gI1;
			hI_current = hI1;
			gI_old = gI0;
			hI_old = hI0;
		} else {
			gE_current = gE0;
			hE_current = hE0;
			gE_old = gE1;
			hE_old = hE1;
			gI_current = gI0;
			hI_current = hI0;
			gI_old = gI1;
			hI_old = hI1;
		}
        high_resolution_clock::time_point vStart = timeNow();
        bool spiked = false;
        for (unsigned int i=0; i<networkSize; i++) {
            if (gI_old[i] > 1.0f || hI_old[i] > 1.0f) {
                printf("%i here to spy, v = %e, gI = %e, hI = %e\n", i, lif[i]->v, gI_old[i], hI_old[i]);
                assert(gI_old[i] == 0.0f);
            }
            lif[i]->v0 = lif[i]->v;
            if (i<nE) {
                gL = gL_E;
                tRef = tRef_E;
            } else {
                gL = gL_I;
                tRef = tRef_I;
            }
            // get a0, b0
            double gE_t, gI_t;
            gE_t = 0.0f;
            #pragma unroll
            for (int ig = 0; ig < ngTypeE; ig++) {
                int gid = networkSize*ig + i;
                gE_t += gE_old[gid];
                gE_current[gid] = gE_old[gid];
                hE_current[gid] = hE_old[gid];
            }
            gI_t = 0.0f;
            #pragma unroll
            for (int ig = 0; ig < ngTypeI; ig++) {
                int gid = networkSize*ig + i;
                gI_t += gI_old[gid];
                gI_current[gid] = gI_old[gid];
                hI_current[gid] = hI_old[gid];
            }
            lif[i]->set_p0(gE_t, gI_t, gL);
			/* === RAND === #ifdef TEST_WITH_MANUAL_FFINPUT */
                nInput[i] = 0;
                if (lTR[i] < dt) {
                    inputTime[i*MAX_FFINPUT_PER_DT] = lTR[i];
                    nInput[i]++;
                    double tmp = lTR[i] + dInput;
                    while (tmp < dt){
                        inputTime[i*MAX_FFINPUT_PER_DT + nInput[i]] = tmp;
                        nInput[i]++;
                        tmp += dInput;
                    }
                    lTR[i] = tmp - dt;
                } else {
                    lTR[i] -= dt;
                }
				inputEvents += nInput[i];
			/* === RAND === 
				#else
					nInput[i] = set_input_time(&(inputTime[i*MAX_FFINPUT_PER_DT]), dt, flatRate, lTR[i], logRand[i], randGen[i]);
					inputEvents += nInput[i];
				#endif
			*/
            /* evolve g to t+dt with ff input only */
            gE_t = 0.0f;
            #pragma unroll
            for (int ig=0; ig<ngTypeE; ig++) {
                unsigned int gid = networkSize*ig + i;
                evolve_g(condE, &(gE_current[gid]), &(hE_current[gid]), &(fE[gid]), &(inputTime[i*MAX_FFINPUT_PER_DT]), nInput[i], dt, dt, ig);
                gE_t += gE_current[gid];
            }
            gI_t = 0.0f; 
            // no feed-forward inhibitory input (setting nInput = 0)
            #pragma unroll
            for (int ig=0; ig<ngTypeI; ig++) {
                unsigned int gid = networkSize*ig + i;
                evolve_g(condI, &(gI_current[gid]), &(hI_current[gid]), &(fI[gid]), inputTime, 0, dt, dt, ig);
                gI_t += gI_current[gid];
            }
            lif[i]->set_p1(gE_t, gI_t, gL);
            // rk2 step
            spikeTrain[i] = cpu_dab(lif[i], 0, dt, dt, dt, tRef, i, gE_t, gI_t);
            if (dt - spikeTrain[i] > EPS) spiked = true;
            v[i] = lif[i]->v;
            assert(v[i]>0.0);
        }
        vTime += static_cast<double>(duration_cast<microseconds>(timeNow()-vStart).count());
            // spike-spike correction
        if (spiked) {
            //printf("spiked:\n");
            high_resolution_clock::time_point sStart = timeNow();
            outputEvents += cpu_ssc(lif, v, gE_old, gI_old, hE_old, hI_old, gE_current, gI_current, hE_current, hI_current, fE, fI, preMat, networkSize, condE, condI, ngTypeE, ngTypeI, inputTime, nInput, spikeTrain, nE, dt);
            sTime += static_cast<double>(duration_cast<microseconds>(timeNow()-vStart).count());
        }
		high_resolution_clock::time_point wStart = timeNow();
        v_file.write((char*)v, networkSize * sizeof(double));
        spike_file.write((char*) spikeTrain, networkSize * sizeof(double));
        gE_file.write((char*)gE_current, networkSize * ngTypeE * sizeof(double));
        gI_file.write((char*)gI_current, networkSize * ngTypeI * sizeof(double));
		wTime += static_cast<double>(duration_cast<microseconds>(timeNow() - wStart).count());
        printf("\r stepping %3.1f%%", 100.0f*float(istep+1)/nstep);
		//printf("gE0 = %f, v = %f \n", gE_current[0], v[0]);
    }
    printf("\n");
    printf("input events rate %fkHz\n", float(inputEvents)/(dt*nstep*networkSize));
    printf("output events rate %fHz\n", float(outputEvents)*1000.0f/(dt*nstep*networkSize));
    auto cpuTime = duration_cast<microseconds>(timeNow()-start).count();
    printf("cpu version time cost: %3.1fms\n", static_cast<double>(cpuTime)/1000.0f);
    printf("compute_V cost: %fms\n", vTime/1000.0f);
    printf("correct_spike cost: %fms\n", sTime/1000.0f);
	printf("writing data to disk cost: %fms\n", wTime/1000.0f);
    /* Cleanup */
    printf("Cleaning up\n");
    int nTimer = 2;
    p_file.write((char*)&nTimer, sizeof(int));
    p_file.write((char*)&vTime, sizeof(double));
    p_file.write((char*)&sTime, sizeof(double));
    
    if (p_file.is_open()) p_file.close();
    if (v_file.is_open()) v_file.close();
    if (spike_file.is_open()) spike_file.close();
    if (gE_file.is_open()) gE_file.close();
    if (gI_file.is_open()) gI_file.close();
    delete []v;
    delete []gE0;
    delete []gI0;
    delete []gE1;
    delete []gI1;
    delete []hE0;
    delete []hI0;
    delete []hE1;
    delete []hI1;
    delete []fE;
    delete []fI;
    delete []preMat;
    delete []spikeTrain;
    /* === RAND === delete []randGen; */
    for (unsigned int i=0; i<networkSize; i++) {
        delete []lif[i];
    }
    delete []lif;

    delete []inputTime;
	delete []logRand;
	delete []lTR;
    delete []nInput;
    printf("Memories freed\n");
}
