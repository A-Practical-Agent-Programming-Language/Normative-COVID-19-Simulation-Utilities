import os

import numpy as np

from analyse_bayesian_optimization import norm_weights, affected_agents_file, omega_c, load_evaluator, \
    max_simulation_date
from classes.ExecutiveOrderOptimizer.EOOptimization import EOOptimization
from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule
from scripts.plot_epicurve_average import EpicurvePlotter

policy = {'EO0_duration': 4.848697127221579, 'EO0_start': 0.8018095678297487, 'EO1_duration': 12.724897486425387,
          'EO1_start': 9.093120679487686, 'EO2_duration': 5.340843994071355, 'EO2_start': 1.488785570621786,
          'EO3_duration': 2.990990679692075, 'EO3_start': 16.874134865788744, 'EO4_duration': 10.598311204870454,
          'EO4_start': 14.386042695453717, 'EO5_duration': 7.499324523476483, 'EO5_start': 0.19397591580917395,
          'EO6_duration': 5.282608442786257, 'EO6_start': 4.041657848046359, 'EO7_duration': 10.186156146913007,
          'EO7_start': 13.88514878454494, 'EO8_duration': 11.394781387008779, 'EO8_start': 1.8825945301682165}

policy = EOOptimization.normalize_params(policy)
directory = "/home/jan/dev/university/2apl/simulation/calibration/output/remote-agent-runs/bayesian-optimization/after-holiday-fix/repeat/default-weights"

ns = NormSchedule(policy, "2020-06-28")
evaluator = load_evaluator(omega_c, norm_weights['default-weights'], affected_agents_file)

targets, infecteds, pcts = list(), list(), list()


for subdir in sorted(os.listdir(directory)):
    if not os.path.isdir(os.path.join(directory, subdir)) or not os.path.exists(os.path.join(directory, subdir, 'epicurve.sim2apl.csv')):
        print("Skipping", directory)
        continue
    last_day_epicurve = EpicurvePlotter.read_epicurve_file(os.path.join(directory, subdir))[0][max_simulation_date]
    population_size = sum([last_day_epicurve[x] for x in last_day_epicurve if type(last_day_epicurve[x]) == int])
    target, infected, _ = evaluator.fitness(directories=[{0: os.path.join(directory, subdir)}], norm_schedule=ns)
    pct = infected / population_size * 100
    targets.append(-1 * target)
    infecteds.append(infected)
    pcts.append(pct)
    print(subdir[subdir.find("run-")-3:], f"{-1*target:.2e}", f"{infected:.0f}", f"{pct:.1f}")

print("")
print(f"Average: {np.average(targets):.2e} (+/- {np.std(targets):1f})\t\t infected: {np.average(infecteds):1f} (+/- {np.std(infecteds):1f}), {np.average(pcts):.1f}\% (+/- {np.std(pcts):.1f}\%)")
