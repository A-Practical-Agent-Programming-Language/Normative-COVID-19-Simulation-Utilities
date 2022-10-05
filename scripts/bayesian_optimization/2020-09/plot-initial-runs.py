import os
import re

from classes.ExecutiveOrderOptimizer.EOEvaluator import EOEvaluator

base_dir = '/home/jan/dev/university/2apl/simulation/calibration/output/remote-agent-runs/bayesian-optimization/2022-09-optimization/initial-policies'
j = os.path.join
exists = lambda *x: os.path.exists(j(*x))

runs = dict()

for run_dir in os.listdir(base_dir):
    match = re.findall(r'policy-(\d+)-run-0', run_dir)
    if match and exists(base_dir, run_dir, 'epicurve.pansim.csv'):
        run = int(match[0])
        infected = EOEvaluator.count_infected_agents([{0: j(base_dir, run_dir)}])
        score = policy_fitness()
        runs[run] = dict(
            path=j(base_dir, run_dir),
            epicurve=j(base_dir, run_dir, 'epicurve.sim2apl.csv'),

            policy_csv=j(base_dir, run_dir, f'exploration-policy-{run}.csv'),
            policy_json=j(base_dir, run_dir, f'exploration-policy-{run}.json')
        )

for i in range(256):
    assert i in runs, f"Missing run {i}"

