import numpy as np

reported_mean_vals = [0.4, 0.38, 0.79, 0.86, 0.8, 0.82, 0.88, 0.74]

"""
To convert probability of infection to our unit time, it looks like their time unit is 0.25 days (=360 minutes), 
as stated in the beginning of the Methods section. 
If we want to figure out the probability, p, of getting infected in m minutes, 
given a probability, q, of getting infected in 360 minutes, 
we have: 
number of time units, u = 360/m; 
the probability of getting infected in u time units = q = 1-(1-p)^u; therefore p = 1-e^(ln(1-q)/u)."""
theirs = 360  # Unit time is 0.25 days = 360 minutes


def convert_unit_time(p_infection, unit_time_source, unit_time_target):
    u = unit_time_source / unit_time_target
    return 1 - np.e ** (np.log(1-p_infection)/u)


for q in reported_mean_vals:
    print(convert_unit_time(q, 360, 5))
