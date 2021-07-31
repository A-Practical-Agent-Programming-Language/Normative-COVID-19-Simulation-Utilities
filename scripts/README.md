This directory contains various scripts for (pre-)processing the data required by the simulation,
or analyzing the output of the simulation.

We refer to the directory in which the calibration script (or, by proxy, the Java code)
puts all the logs and outputs as the *simulation output directory*.

The *run configuration* refers to the parameters with which the simulation was set-up, and
include all calibrated parameters (i.e., the disease model, the mean of the beta 
distributions for trust in the government of both liberal and conservative voters, 
the fatigue factor, and the simulation timestep at which the fatigue starts).

### Calibration results
- **plot_epicurve.py**: Takes one or more simulation output directories with the same run 
  configuration and plots the epicurve with the target curve (i.e., the actual observed
  number of cases in the simulated counties). Optionally also creates a LaTeX table used by
  Tikz plots.
  
- **plot_mobility_average.py**: Takes one or more simulation output directories with the same
run configuration and plots the (average and standard deviation) mobility for each simulated
  county along the actually recorded mobility.
  

