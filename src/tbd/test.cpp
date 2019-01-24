#include "qromb.h"
#include "func.h"
#include "initialize.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>

int main(){
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::minstd_rand0 generator(seed);
    std::ofstream integrand_points, v_data, g_data, cpu_time;
    integrand_points.open("Integrand-Data.bin", std::ios::out|std::ios::binary);
    v_data.open("v-Data.bin", std::ios::out|std::ios::binary);
    g_data.open("g-Data.bin", std::ios::out|std::ios::binary);
    cpu_time.open("cpu-time.bin", std::ios::out|std::ios::binary);
    using milli = std::chrono::milliseconds;

    float_point f = 100.0;
    float_point g0 = 0.0;
    float_point h0 = 0.0;

    float_point tau_e1 = 0.001;
    float_point tau_e2 = 0.003;

    float_point tau_i1 = 0.001;
    float_point tau_i2 = 0.005;
    float_point eE = 14.0/3.0;
    float_point eI = -2.0/3.0;
    float_point gL = 50.0;
    float_point eL = 0.0;
    float_point eT = 1.0;


    int ntype = 2;

    Rem** r = new Rem*[ntype];
    //r[0] = new Rem(g0);
    //r[1] = new RemAlpha(g0, h0);
    //
    //r[0] = new Rem(g0);
    //r[1] = new Rem(g0);
    //
    r[0] = new RemAlpha(g0, h0);
    r[1] = new RemAlpha(g0, h0);

    CondP** cP = new CondP*[ntype];
    //cP[0] = new CondP(f, tau_e2, eE);
    //cP[1] = new CondAlphaP(f, tau_e1, tau_e2, eE);
    //
    //cP[0] = new CondP(f, tau_e2, eE);
    //cP[1] = new CondP(f, tau_e2, eE);
    //
    cP[0] = new CondAlphaP(f, tau_e1, tau_e2, eE);
    cP[1] = new CondAlphaP(f*7.0, tau_i1, tau_i2, eI);

    Cond **cond = new Cond*[ntype];
    Cond **cond_rk2 = new Cond*[ntype];
    //cond[0] = new CondExp(*cP[0],*r[0]);
    //cond[1] = new CondAlpha(*static_cast<CondAlphaP*>(cP[1]), *static_cast<RemAlpha*>(r[1]));
    //cond_rk2[0] = new CondExp(*cP[0],*r[0]);
    //cond_rk2[1] = new CondAlpha(*static_cast<CondAlphaP*>(cP[1]), *static_cast<RemAlpha*>(r[1]));
    //
    //cond[0] = new CondExp(*cP[0],*r[0]);
    //cond[1] = new CondExp(*cP[1],*r[1]);
    //cond_rk2[0] = new CondExp(*cP[0],*r[0]);
    //cond_rk2[1] = new CondExp(*cP[1],*r[1]);
    //
    cond[0] = new CondAlpha(*static_cast<CondAlphaP*>(cP[0]),*static_cast<RemAlpha*>(r[0]));
    cond[1] = new CondAlpha(*static_cast<CondAlphaP*>(cP[1]),*static_cast<RemAlpha*>(r[1]));
    cond_rk2[0] = new CondAlpha(*static_cast<CondAlphaP*>(cP[0]), *static_cast<RemAlpha*>(r[0]));
    cond_rk2[1] = new CondAlpha(*static_cast<CondAlphaP*>(cP[1]), *static_cast<RemAlpha*>(r[1]));


    int verbose = 1;
    int verdVcomp;
    int verInput;
    int verOut;
    float_point evalt = tau_e2;
    float_point evalt2 = evalt; 
    bool analyticSimpleTest = false;
    if (analyticSimpleTest) {
        for (int i =0; i< ntype; i++) {
            Func1Cond_g fCond_g(cond[i]);
            Func1Cond_dg fCond_dg(cond[i]);
            Func1Cond_gr fCond_gr(cond[i]);
            Func1Cond_dgr fCond_dgr(cond[i]);
            cond[i]->display();
            std::cout << "g:" << i << "\n";
            std::cout << "  " << qromb(&fCond_g, 4, evalt, verbose) << "\n";
            cond[i]->set_dtTempVar(evalt);
            std::cout << "  analytic: " << cond[i]->eval_ig(evalt) << "\n";

            std::cout << "  " << qromb(&fCond_g, 3, evalt2, verbose) << "\n";
            cond[i]->set_dtTempVar(evalt2);
            std::cout << "  analytic: " << cond[i]->eval_ig(evalt2) << "\n";

            std::cout << "dg:" << i << "\n";
            cond[i]->set_dtTempVar(0.0);
            float_point a = cond[i]->eval_g(0.0);

            std::cout << "  " << qromb(&fCond_dg, 4, evalt, verbose) << "\n";
            cond[i]->set_dtTempVar(evalt);
            std::cout << "  analytic: " << cond[i]->eval_g(evalt) - a << "\n";

            std::cout << "  " << qromb(&fCond_dg, 3, evalt2, verbose) << "\n";
            cond[i]->set_dtTempVar(evalt2);
            std::cout << "  analytic: " << cond[i]->eval_g(evalt2) - a << "\n";

            std::cout << "g_rem:" << i << "\n";
            std::cout << "  " << qromb(&fCond_gr, 4, evalt, verbose) << "\n";
            cond[i]->set_dtTempVar(evalt);
            std::cout << "  analytic: " << cond[i]->eval_ig_rem(evalt) << "\n";

            std::cout << "  " << qromb(&fCond_gr, 3, evalt2, verbose) << "\n";
            cond[i]->set_dtTempVar(evalt2);
            std::cout << "  analytic: " << cond[i]->eval_ig_rem(evalt2) << "\n";

            std::cout << "dg_rem:" << i << "\n";
            cond[i]->set_dtTempVar(0.0);
            a = cond[i]->eval_g_rem(0.0);

            std::cout << "  " << qromb(&fCond_dgr, 4, evalt, verbose) << "\n";
            cond[i]->set_dtTempVar(evalt);
            std::cout << "  analytic: " << cond[i]->eval_g_rem(evalt) - a << "\n";

            std::cout << "  " << qromb(&fCond_dgr, 3, evalt2, verbose) << "\n";
            cond[i]->set_dtTempVar(evalt2);
            std::cout << "  analytic: " << cond[i]->eval_g_rem(evalt2) - a << "\n";
        }
    }
    
    int* nInput = new int[ntype];
    nInput[0] = 250;
    nInput[1] = 25;
    float_point runTime = tau_e1*100;
    float_point tInEnd = tau_e1*100;
    ntype = 2;
    int nNYU = 6;
    int nrk = 6;
    float_point truth_dt = 1e-8;
    float_point* NYU_dt = new float_point[nNYU];
    float_point* iter = new float_point[nNYU];
    NYU_dt[0] = 0.004;
    NYU_dt[1] = 0.002;
    NYU_dt[2] = 0.001;
    NYU_dt[3] = 0.0005;
    NYU_dt[4] = 0.00025;
    NYU_dt[5] = truth_dt;
    //NYU_dt[3] = 0.000001;
    iter[0] = 2;
    iter[1] = 2;
    iter[2] = 2;
    iter[3] = 2;
    iter[4] = 2;
    iter[5] = 2;
    //iter[2] = 3;
    //iter[3] = 15;

    float_point* rk2_dt = new float_point[nrk]; 
    rk2_dt[0] = 0.004;
    rk2_dt[1] = 0.002;
    rk2_dt[2] = 0.001;
    rk2_dt[3] = 0.0005;
    rk2_dt[4] = 0.00025;
    rk2_dt[nrk-1] = truth_dt;


    float_point **tIn = new float_point*[ntype];
    for (int i=0; i<ntype; i++) {
        tIn[i] = new float_point[nInput[i]];
        for (int j=0; j<nInput[i]; j++) {
            tIn[i][j] = tInEnd/nInput[i]*(j+distribution(generator)); //+ tInEnd/5.0;
            //tIn[i][j] = tInEnd/nInput[i]*(j+0.5*i); //+ tInEnd/5.0;
        }
    }

    float_point Vs, GsEs, Gs;
    verbose = 0;
    verdVcomp = 1;
    verInput = 0;
    verOut = 0;
    int verInt = 0;
    float_point **ctIn = new float_point*[ntype];
    int* nIn = new int[ntype];
    int* iIn = new int[ntype];
    for (int i=0; i < ntype; i++) {
        iIn[i] = 0;
    }
    mNYU_vsDecayInt test(ntype,cond,gL,eL); 
    std::vector<float_point> nevals;
    float_point* V_0 = new float_point[nNYU];
    float_point Vs_0, V_old;
    std::vector<std::chrono::duration<double>> NYU_time;
    int* NYU_sp = new int[nNYU];
    v_data.write((char*)&nNYU, sizeof(int));
    for (int iNYU = 0; iNYU < nNYU; iNYU ++ ) {
        float_point dt = NYU_dt[iNYU];
        NYU_sp[iNYU] = 0;
        int nt = round(runTime/dt);
        int evalPerInt = pow(2,iter[iNYU]-1) + 1;
        nevals.push_back(evalPerInt*nt);
        integrand_points.write((char*)&evalPerInt, sizeof(int));
        integrand_points.write((char*)&nt, sizeof(int));
        for (int i=0; i<ntype; i++) {
            cond[i]->reset(r[i]);
            iIn[i] = 0;
        }
        if (verbose > 0) {
            std::cout << " NYU inter: " << iter[iNYU] << ", dt: " << dt <<  "========================================\n";
        }
        Vs_0 = 0.0;
        V_old = 0.0;
        V_0[iNYU] = 0.0;
        if (verbose > 0) {
            std::cout << "V0: " << V_0[iNYU] << "\n";
        }
        v_data.write((char*)&nt, sizeof(int));
        v_data.write((char*)&dt, sizeof(float_point));
        v_data.write((char*)&(V_0[iNYU]), sizeof(float_point));
        auto start = std::chrono::high_resolution_clock::now();
        double g_avg = 0.0f;
        double v_avg = 0.0f;
        for  (int it = 0; it < nt; it++) {
            //test.display();
            //test.book_lookup();
            for (int i=0; i < ntype; i++) {
                nIn[i] = 0;
                for (int j = iIn[i]; j<nInput[i]; j++) {
                    if (tIn[i][j] > dt*(it+1)-1e-15) {
                        if (verbose > verInput) {
                            if (j>iIn[i]) {
                                std::cout << "; ";
                            }
                        }
                        nIn[i] = j-iIn[i];
                        break;
                    } else {
                        if (verbose > verInput) {
                            if (j==iIn[i]) {
                                std::cout << "type " << i << " input: ";
                            }
                            std::cout << tIn[i][j];
                            if (j>iIn[i]) {
                                std::cout << ", ";
                            }
                        }
                        if (j == nInput[i]-1) {
                            nIn[i] = nInput[i]-iIn[i];
                        }
                    }
                }
                ctIn[i] = new float_point[nIn[i]];
                for (int j = 0; j<nIn[i]; j++) {
                    ctIn[i][j] = tIn[i][iIn[i]+j]-it*dt;
                    if (verbose > verInput) {
                        if (j==0) {
                            std::cout << "relative to time step: ";
                        }
                        std::cout << ctIn[i][j];
                        if (j == nIn[i] - 1) {
                            std::cout << "\n";
                        } else {
                            std::cout << ", ";
                        }
                    }
                }
            }
            test.step(dt, ctIn, nIn); 
            float_point GsEs = gL*eL, Gs = gL;
            for (int i=0; i<ntype; i++) {
                Gs += cond[i]->g_dt;
                GsEs += cond[i]->g_dt*cond[i]->get_e();
            }
            float_point Vs = GsEs/Gs;
            float_point analytic_V = Vs + test.integrated_factor*(V_0[iNYU]-Vs_0); 
            float_point integral = qromb(&test, iter[iNYU], dt, verInt, integrand_points);
            if (verbose > verdVcomp) {
                std::cout << "vs{i} + exp(-GsInt)*dV_Vs{i-1} - integral: " <<  Vs << "+"<< test.integrated_factor << "(" << V_0[iNYU] - Vs_0 << ") - " << integral << "\n";
                std::cout << "g rem: ";
                for (int ic=0; ic<ntype; ic++) {
                    std::cout << cond[ic]->g_rem << ", ";
                }
                std::cout << "rem_0: ";
                for (int ic=0; ic<ntype; ic++) {
                    std::cout << cond[ic]->get_last_g() << ", ";
                }
                test.eval(0);
                std::cout << "g_t0 " << test.Gs << "\n";
            }
            V_0[iNYU] = analytic_V - integral;
            if (V_0[iNYU] > eT) {
                float_point dtsp = (eT-V_old)/(V_0[iNYU]-V_old)*dt;
                //std::cout << "dtsp " << dtsp << "\n";
                V_0[iNYU] = eL;
                int* nTemp = new int[ntype];
                for (int i=0; i<ntype; i++) {
                    nTemp[i] = nIn[i];
                    for (int j = iIn[i]+nIn[i]-1; j>iIn[i]; j--) {
                        if (tIn[i][j] < dt*(it)+dtsp) {
                            nTemp[i] = j-iIn[i]+1;
                            break;
                        } else {
                            if (j == iIn[i]) {
                                nTemp[i] = 0;
                            }
                        }
                    }
                    cond[i]->advance(ctIn[i], nTemp[i], dtsp);
                    cond[i]->next_rem();
                }
                float_point** dtIn = new float_point*[ntype];
                for (int i=0; i<ntype; i++) {
                    iIn[i] = iIn[i]+nTemp[i];
                    nIn[i] = nIn[i]-nTemp[i];
                    for (int j=0; j<nIn[i]; j++) {
                        ctIn[i][nTemp[i]+j] = ctIn[i][nTemp[i]+j] - dtsp;
                    }
                    dtIn[i] = &ctIn[i][nTemp[i]];
                }
                test.step(dt-dtsp, dtIn, nIn); 
                GsEs = gL*eL, Gs = gL;
                for (int i=0; i<ntype; i++) {
                    Gs += cond[i]->g_dt;
                    GsEs += cond[i]->g_dt*cond[i]->get_e();
                }
                Vs = GsEs/Gs;
                analytic_V = Vs + test.integrated_factor*(V_0[iNYU]-Vs_0); 
                integral = qromb(&test, iter[iNYU], dt-dtsp, verInt);
                V_0[iNYU] = analytic_V - integral;
                delete []dtIn;
                delete []nTemp;
                NYU_sp[iNYU] = NYU_sp[iNYU] + 1;
            }
            if (verbose > verOut) {
                std::cout << it*dt << ": " << Gs << ", " << Vs << ", " << V_0[iNYU] << "\n";
            }
            g_avg += Gs;
            v_avg += V_0[iNYU];
            v_data.write((char*)&(V_0[iNYU]), sizeof(float_point));
            g_data.write((char*)&Gs, sizeof(float_point));
            for (int i=0; i < ntype; i++) {
                iIn[i] = iIn[i] + nIn[i];
                delete []ctIn[i];
            }
            Vs_0 = Vs;
            V_old = V_0[iNYU];
            test.finish_step(); 
            if (verbose > verInput && it*dt > tInEnd) {
                std::cout << "no Input\n";
            }
        }
        auto finish = std::chrono::high_resolution_clock::now();
        NYU_time.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(finish - start));
        std::cout << "average g: " << g_avg/nt << "v: " << v_avg/nt << "\n";
    }

    verbose = 0;
    verInput = 0;
    verOut = 0;
    float_point* v_0 = new float_point[nrk];
    std::vector<std::chrono::duration<double>> rk2_time;
    int* rk2_sp = new int[nrk];
    v_data.write((char*)&nrk, sizeof(int));
    for (int irk = 0; irk < nrk; irk++) {
        rk2_sp[irk] = 0;
        float_point dt = rk2_dt[irk];
        int nt = round(runTime/dt);
        v_0[irk] = 0.0;
        
        if (verbose > 0) {
            std::cout << " rk2 dt: " << dt << "========================================\n";
        }

        for (int i=0; i < ntype; i++) {
            cond_rk2[i]->reset(r[i]);
            iIn[i] = 0;
        }
        GsEs = gL*eL, Gs = gL;
        for (int i=0; i<ntype; i++) {
            Gs += cond_rk2[i]->get_last_g();
            GsEs += cond_rk2[i]->get_last_g()*cond_rk2[i]->get_e();
        }
        //std::cout << "GsEs: " << GsEs << "Gs: " << Gs << "\n";
        
        double g_avg = 0.0f;
        double v_avg = 0.0f;
        v_data.write((char*)&nt, sizeof(int));
        v_data.write((char*)&dt, sizeof(float_point));
        v_data.write((char*)&(v_0[irk]), sizeof(float_point));
        auto start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it<nt; it++) {
            for (int i=0; i < ntype; i++) {
                nIn[i] = 0;
                for (int j = iIn[i]; j<nInput[i]; j++) {
                    if (tIn[i][j] > dt*(it+1)-1e-15) {
                        if (verbose > verInput) {
                            if (j>iIn[i]) {
                                std::cout << "; ";
                            }
                        }
                        nIn[i] = j-iIn[i];
                        break;
                    } else {
                        if (verbose > verInput) {
                            if (j==iIn[i]) {
                                std::cout << "type " << i << " input: ";
                            }
                            std::cout << tIn[i][j];
                            if (j>iIn[i]) {
                                std::cout << ", ";
                            }
                        }
                        if (j == nInput[i]-1) {
                            nIn[i] = nInput[i]-iIn[i];
                        }
                    }
                }
                ctIn[i] = new float_point[nIn[i]];
                for (int j = 0; j<nIn[i]; j++) {
                    ctIn[i][j] = dt-(tIn[i][iIn[i]+j]-it*dt);
                    if (verbose > verInput) {
                        if (j==0) {
                            std::cout << "decay duration in the time step: ";
                        }
                        std::cout << ctIn[i][j];
                        if (j == nIn[i] - 1) {
                            std::cout << "\n";
                        } else {
                            std::cout << ", ";
                        }
                    }
                }
            }
            float_point fk1 = -Gs*v_0[irk] + GsEs;
            float_point GsEs_old = GsEs;
            float_point Gs_old = Gs;
            GsEs = gL*eL, Gs = gL;
            for (int i=0; i<ntype; i++) {
                cond_rk2[i]->book_keeping(ctIn[i], nIn[i], dt);
                Gs += cond_rk2[i]->g_dt;
                GsEs += cond_rk2[i]->g_dt*cond_rk2[i]->get_e(); 
            } 
            float_point v1 = v_0[irk] + dt*fk1;
            float_point fk2 = -Gs*v1 + GsEs;
            float_point v_old = v_0[irk];
            //std::cout << "fk1: " << fk1 << "fk2: " << fk2 << "\n";
            v_0[irk] = v_0[irk] + dt/2.0*(fk1+fk2);
            if (v_0[irk] > eT) {
                float_point dtsp = (eT-v_old)/(v_0[irk]-v_old)*dt;
                //std::cout << "GsEs_old " << GsEs_old << ", Gs_old " << Gs_old << "\n";
                //std::cout << "GsEs " << GsEs << ", Gs " << Gs << "\n";
                float_point vTemp = (eL-dtsp*(GsEs_old + GsEs - Gs*GsEs_old*dt)/2.0)/(1.0+dtsp*(-Gs_old-Gs+Gs*Gs_old*dt)/2.0);
                //std::cout << "dtsp " << dtsp << ", vn0 " << vTemp << "\n";
                fk1 = -Gs_old*vTemp + GsEs_old;
                v1 = vTemp + dt*fk1;
                fk2 = -Gs*v1 + GsEs;
                v_0[irk] = vTemp + dt*(fk1+fk2)/2.0;
                rk2_sp[irk] = rk2_sp[irk] + 1;
            }
            if (verbose > verOut) {
                std::cout << it*dt << ": " << Gs << ", " << GsEs/Gs << ", "  << v_0[irk] << "\n";
            }
            v_data.write((char*)&(v_0[irk]), sizeof(float_point));
            g_data.write((char*)&Gs, sizeof(float_point));
            for (int i=0; i<ntype; i++) {
                cond_rk2[i]->next_rem();
            } 
            for (int i=0; i < ntype; i++) {
                iIn[i] = iIn[i] + nIn[i];
                delete []ctIn[i];
            }
            if (verbose > verInput && it*dt > tInEnd) {
                std::cout << "no Input\n";
            }
            g_avg += Gs;
            v_avg += v_0[irk];
        }
        auto finish = std::chrono::high_resolution_clock::now();
        rk2_time.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(finish - start));
        std::cout << "average g: " << g_avg/nt << "v: " << v_avg/nt << "\n";
    }

    std::cout << "              ";
    for (int irk = 0; irk<nrk; irk++) {
        if (irk==nrk-1) {
            std::cout << "truth_rk-";
        } else {
            std::cout << "rk-";
        }
        std::cout << rk2_dt[irk] << "              ";
    }
    for (int iNYU = 0; iNYU<nNYU; iNYU++) {
        std::cout << "NYU-" << NYU_dt[iNYU] << "-" << iter[iNYU];
        if (iNYU == nNYU-1) {
            std::cout << "\n";
        }
    }

    std::cout << "last step:  " << std::setprecision(8);
    for (int irk = 0; irk<nrk; irk++) {
        std::cout << v_0[irk] << "              ";
    }
    for (int iNYU = 0; iNYU<nNYU; iNYU++) {
        std::cout << V_0[iNYU] << "              ";
        if (iNYU == nNYU-1) {
            std::cout << "\n";
        }
    }

    std::cout << "evals     :  ";
    for (int irk = 0; irk<nrk; irk++) {
        std::cout << round(runTime/rk2_dt[irk]) << "              ";
    }
    for (int iNYU = 0; iNYU<nNYU; iNYU++) {
        std::cout << nevals[iNYU] << "              ";
        if (iNYU == nNYU-1) {
            std::cout << "\n";
        }
    }

    std::cout << "time     :  ";
    for (int irk = 0; irk<nrk; irk++) {
        float_point ct = rk2_time[irk].count();
        std::cout << ct << "              ";
        cpu_time.write((char*)&ct, sizeof(double));
    }
    for (int iNYU = 0; iNYU<nNYU; iNYU++) {
        float_point ct = NYU_time[iNYU].count();
        std::cout << ct << "              ";
        cpu_time.write((char*)&ct, sizeof(double));
        if (iNYU == nNYU-1) {
            std::cout << "\n";
        }
    }

    std::cout << "spikes     :  ";
    for (int irk = 0; irk<nrk; irk++) {
        std::cout << rk2_sp[irk] << "              ";
    }
    for (int iNYU = 0; iNYU<nNYU; iNYU++) {
        std::cout << NYU_sp[iNYU] << "              ";
        if (iNYU == nNYU-1) {
            std::cout << "\n";
        }
    }


        
    for (int i=0; i<ntype; i++) {
        delete []tIn[i];
        delete r[i];
        delete cP[i];
        delete cond[i];
        delete cond_rk2[i];
    }
    v_data.close();
    g_data.close();
    cpu_time.close();
    integrand_points.close();
    delete []NYU_sp;
    delete []rk2_sp;
    delete []rk2_dt;
    delete []NYU_dt;
    delete []iter;
    delete []v_0;
    delete []V_0;
    delete []r;
    delete []cP;
    delete []cond;
    delete []cond_rk2;
    delete []tIn;
    delete []ctIn;
    delete []nIn;
    delete []iIn;
    delete []nInput;
    return 0;
}
