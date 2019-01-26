double cpu_dab(cpu_LIF* lif, double dt, double tRef, unsigned int id, double gE, double gI) {
    lif->tsp = dt;
    lif->correctMe = true;
    lif->spikeCount = 0;
    // not in refractory period
    if (lif->tBack < dt) {
        // return from refractory period
        if (lif->tBack > 0.0f) {
            lif->compute_pseudo_v0(dt);
            lif->tBack = -1.0f;
        }
        lif->runge_kutta_2(dt);
        if (lif->v > vT) {
            // crossed threshold
            if (lif->v > vE) {
				printf("#%i something is off gE = %f, gI = %f, v = %f\n", id, gE, gI, lif->v);
                lif->v = vE;
            }

            lif->tsp = lif->compute_spike_time(dt); 
            // dabbing not commiting, doest not reset v or recored tBack, TBD by spike correction.
        }
    } else {
        // during refractory period
        lif->reset_v(); 
        lif->tBack -= dt;
        lif->correctMe = false;
    } 
    return lif->tsp;
}
