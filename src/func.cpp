#include "func.h"

// Rangan_Cai_Integrand begin 
    Rangan_Cai_Integrand::Rangan_Cai_Integrand(int _ntype, Cond**_cond, float_point _gL, float_point _eL): ntype(_ntype), gL(_gL), eL(_eL) {
        // create conductance shapes and pass pointers to cond
        cond = new Cond*[ntype];
        for (int i=0; i<ntype; i++) {
            cond[i] = _cond[i];
        }
        tIn = new float_point*[ntype];
        dtIn = new float_point*[ntype];
    }
    
    float_point Rangan_Cai_Integrand::eval(float_point t) {
        GsInt = gL*(dt-t);
        dGs = 0.0;
        Gs = gL;
        GsEs = gL*eL;
        dGsEs = 0.0;
        for (int i=0; i<ntype; i++) {
            float_point dGsTotal = 0.0;
            float_point GsTotal = 0.0;
            for (int j=0; j<n[i]; j++) {
                cond[i]->eval(t-tIn[i][j]);

                dGsTotal += cond[i]->dg;
                GsTotal += cond[i]->g;
                GsInt += cond[i]->condInt[j] - cond[i]->ig;
            }
            cond[i]->eval_rem(t);

            dGsTotal += cond[i]->dg_rem;
            GsTotal += cond[i]->g_rem;
            GsInt += cond[i]->condRemInt - cond[i]->ig_rem;

            dGs += dGsTotal;
            Gs += GsTotal;
            dGsEs += dGsTotal*cond[i]->get_e();
            GsEs += GsTotal*cond[i]->get_e();
            //std::cout << "  Es: " << cond[i]->get_e() << "\n";
        }
        integrating_factor = exp(-GsInt);
        dVs = (dGsEs - dGs*GsEs/Gs)/Gs;
        //std::cout << "=============\n";
        //std::cout << "  Gs: " << Gs << "\n";
        //std::cout << "  dGs: " << dGs << "\n";
        //std::cout << "  GsInt: " << GsInt << "\n";
        //std::cout << "  GsEs: " << GsEs << "\n";
        //std::cout << "  dGsEs: " << dGsEs << "\n";
        //std::cout << "  efactor: " << integrating_factor << ", ";
        //std::cout << "  dVs: " << dVs << "; ";
        //std::cout << "=============\n";
        return integrating_factor*dVs;
    }
    
    void Rangan_Cai_Integrand::step(float_point _dt, float_point **_tIn, int *_n) {
        n = _n;
        dt = _dt;
        for (int i=0; i<ntype; i++) {
            tIn[i] = _tIn[i];
            dtIn[i] = new float_point[n[i]];
            for (int j=0; j<n[i]; j++) {
                dtIn[i][j] = dt - tIn[i][j];
            }
        }
        book_keeping();
    }

    void Rangan_Cai_Integrand::finish_step() {
        for (int i=0; i<ntype; i++) {
            cond[i]->next_rem();
            delete []dtIn[i];
        }
    }

    void Rangan_Cai_Integrand::display() {
        std::cout << " The form of integrand (Rangan Cai 2007): int(exp(-G_s))*dV_s/ds\n";
        std::cout << ", where G_s = sum(G)";
        std::cout << " the integrand contains the following conductance types (G):\n";
        for (int i=0; i<ntype; i++) {
            cond[i]->display();
            std::cout << " input time: ";
            for (int j=0; j<n[i]; j++) {
                std::cout << tIn[i][j];
                if (j==n[i]-1) {
                    std::cout << "\n";
                } else {
                    std::cout << ", ";
                }
            }
        }
    }

    void Rangan_Cai_Integrand::book_keeping() {
        GsInt = gL*dt;
        for (int i=0; i<ntype; i++) {
            cond[i]->book_keeping(dtIn[i],n[i],dt);
            for (int j=0; j<n[i]; j++) {
                GsInt += cond[i]->condInt[j];
            }
            GsInt += cond[i]->condRemInt;
        }
        integrated_factor = exp(-GsInt);
    }

    void Rangan_Cai_Integrand::reset_dt(int *_n, float_point _dt) {
        n = _n;
        dt = _dt;
        for (int i=0; i<ntype; i++) {
            for (int j=0; j<n[i]; j++) {
                dtIn[i][j] = dt - tIn[i][j];
            }
        }
        book_keeping();
    }
    
    void Rangan_Cai_Integrand::book_lookup() {
        for (int i=0; i<ntype; i++) {
            std::cout << "inputs: ";
            for (int j=0; j<n[i]; j++) {
                std::cout << cond[i]->condInt[j];
                if (j==n[i]-1) {
                    std::cout << "\n";
                } else {
                    std::cout << ", ";
                }
            }
            std::cout << "remain: " << cond[i]->condRemInt << "\n";
        }
    }

    Rangan_Cai_Integrand::~Rangan_Cai_Integrand() {
        delete []tIn;
        delete []dtIn;
        delete []cond;
    }
// Rangan_Cai_Integrand end

// Conductance begin
    Conductance::Conductance() {
        condInt = NULL;
    }

    void Conductance::book_keeping(float_point *dtIn, int n, float_point dt) {
        delete []condInt;
        condInt = new float_point[n];
        g_dt = 0.0;
        evolve_init(); // initialize extra variable
        for (int i=0; i<n; i++) {
            set_dtTempVar(dtIn[i]);
            condInt[i] = eval_ig(dtIn[i]);
            evolve_dt(dtIn[i]);  // evolve extra variable
            g_dt += eval_g(dtIn[i]);
        }
        set_dtTempVar(dt);
        condRemInt = eval_ig_rem(dt);
        evolve_dt_rem(dt); // evolve extra variable
        g_dt += eval_g_rem(dt);
    }

    void Conductance::advance(float_point *dtIn, int n, float_point dt) {
        g_dt = 0.0;
        evolve_init(); // initialize extra variable
        for (int i=0; i<n; i++) {
            float_point dur = dt-dtIn[i];
            set_dtTempVar(dur);
            evolve_dt(dur);  // evolve extra variable
            g_dt += eval_g(dur);
        }
        set_dtTempVar(dt);
        evolve_dt_rem(dt); // evolve extra variable
        g_dt += eval_g_rem(dt);
    }
    void Conductance::eval(float_point t) {
        if (t>=0) {
            set_dtTempVar(t);
            dg = eval_dg(t);
            g  = eval_g(t);
            ig = eval_ig(t);
        } else {
            dg = 0.0;
            g  = 0.0;
            ig = 0.0;
        }
    }

    void Conductance::eval_rem(float_point t) {
        set_dtTempVar(t);
        dg_rem = eval_dg_rem(t);
        g_rem  = eval_g_rem(t);
        ig_rem = eval_ig_rem(t);
    }

    Conductance::~Conductance() {
        delete []condInt;
    }
// Conductance end

// Conductance_Exp begin
    Conductance_Exp::Conductance_Exp(CondP _p, Rem _r): p(_p), r(_r) {
        cond_id = 1;
    }
    inline void Conductance_Exp::set_dtTempVar(float_point t) {
        etd = exp(-t/p.tau_decay);
    }
    
    inline float_point Conductance_Exp::eval_dg(float_point t) {
        return -p.f/p.tau_decay*etd;
    }
    inline float_point Conductance_Exp::eval_g(float_point t) {
        return p.f*etd;
    }
    inline float_point Conductance_Exp::eval_ig(float_point t) {
        return p.f*p.tau_decay*(1-etd);
    }

    inline float_point Conductance_Exp::eval_dg_rem(float_point t) {
        return -r.g0/p.tau_decay*etd;
    }
    inline float_point Conductance_Exp::eval_g_rem(float_point t) {
        return r.g0*etd;
    }
    inline float_point Conductance_Exp::eval_ig_rem(float_point t) {
        return r.g0*p.tau_decay*(1-etd);
    }

    void Conductance_Exp::next_rem() {
        r.g0 = g_dt;
    }
    void Conductance_Exp::reset(Rem *_r) {
        r.g0 = _r->g0;
    }

    float_point Conductance_Exp::get_e() {
        return p.e;
    }
    float_point Conductance_Exp::get_last_g() {
        return r.g0;
    }
    void Conductance_Exp::display() {
        std::cout << "g(t) = f/tau*exp(-t/tau), t>=0\n";
        std::cout << "f = " << p.f << "\n";
        std::cout << "tau = " << p.tau_decay << "\n";
    }
// Conductance_Exp end 

// Conductance_Alpha begin
    Conductance_Alpha::Conductance_Alpha(CondAlphaP _p, RemAlpha _r): p(_p), r(_r) {
        cond_id = 2;
        dtau = p.tau_decay - p.tau_rise;
        rconst = p.tau_decay/dtau;
        f_rconst = p.f*rconst;
        h0_rconst = r.h0*rconst;
    }
    inline void Conductance_Alpha::set_dtTempVar(float_point t) {
        etd = exp(-t/p.tau_decay);
        etr = exp(-t/p.tau_rise);
    }
    
    inline float_point Conductance_Alpha::eval_dg(float_point t) {
        return f_rconst*(etr/p.tau_rise - etd/p.tau_decay);
    }
    inline float_point Conductance_Alpha::eval_g(float_point t) {
        return f_rconst*(etd-etr);
    }
    inline float_point Conductance_Alpha::eval_ig(float_point t) {
        return f_rconst*(dtau-(etd*p.tau_decay - etr*p.tau_rise)); 
    }

    inline float_point Conductance_Alpha::eval_dg_rem(float_point t) {
        return -r.g0*etd/p.tau_decay    + h0_rconst*(etr/p.tau_rise-etd/p.tau_decay);
    }
    inline float_point Conductance_Alpha::eval_g_rem(float_point t) {
        return r.g0*etd                 + h0_rconst*(etd-etr);
    }
    inline float_point Conductance_Alpha::eval_ig_rem(float_point t) {
        return r.g0*p.tau_decay*(1-etd) + h0_rconst*(dtau-(etd*p.tau_decay-etr*p.tau_rise));
    }

    inline void Conductance_Alpha::evolve_init() {
        h_dt = 0.0;
    }
    inline void Conductance_Alpha::evolve_dt(float_point t) {
        h_dt += p.f*etr;
    }

    inline void Conductance_Alpha::evolve_dt_rem(float_point t) {
        h_dt += r.h0*etr;
    }

    float_point Conductance_Alpha::get_e() {
        return p.e;
    }
    float_point Conductance_Alpha::get_last_g() {
        return r.g0;
    }
    void Conductance_Alpha::next_rem() {
        r.g0 = g_dt;
        r.h0 = h_dt;
        h0_rconst = r.h0*rconst;
    }
    void Conductance_Alpha::reset(Rem *_r) {
        r.g0 = static_cast<RemAlpha*>(_r)->g0;
        r.h0 = static_cast<RemAlpha*>(_r)->h0;
        h0_rconst = r.h0*rconst;
    }
    
    void Conductance_Alpha::display() {
        std::cout << "g(t) = f*(exp(-t/tau_d) - exp(-t/tau_r))/(tau_d-tau_r), t>=0\n";
        std::cout << "f = " << p.f << "\n";
        std::cout << "tau_r = " << p.tau_rise << "\n";
        std::cout << "tau_d = " << p.tau_decay << "\n";
    }
// Conductance_Alpha end
