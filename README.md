For simulation results of the full model in the paper "Uncovering potential effects of spontaneous waves on synaptic development: The visual system as a model" by Jennifer Crodell & Wei P. Dai, PRX Life 2026
1. enumerate Compile with `src/compile-nonhpc`
2. Setup spontaneous waves use `ext_input.ipynb`
3. Use `src/wave_II.cfg`, `src/wave_III.cfg` or `src/wave_concat.cfg` (the partially set configuration files) to set the parameters
4. Run with the batch script `src/lFF-hd_restricted-random-trials`
5. Produce additional figures with `src/plot_random_trials_dist.py`
