import os
import re
from typing import Dict, List

from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule
from analyse_bayesian_optimization import load_evaluator, omega_c, affected_agents_file
from utility.utility import get_project_root

# dir_to_check = "/home/jan/dev/university/2apl/simulation/calibration/output/remote-agent-runs/still-working-economy"
dir_to_check = "/output/remote-agent-runs/bayesian-optimization/after-holiday-fix/repeat/default-weights"

directories: List[Dict[str, any]] = list()
for subdir in os.listdir(dir_to_check):
    abs_dir_to_check = os.path.join(dir_to_check, subdir)
    if os.path.isdir(abs_dir_to_check) and os.path.exists(os.path.join(abs_dir_to_check, 'epicurve.pansim.csv')):
        directories.append({'absdir': [{0: abs_dir_to_check}], 'dir': subdir, 'policy': None, 'score': None})

evaluator = load_evaluator(omega_c, os.path.join(get_project_root(), '../../external', 'norm_weights.csv'), affected_agents_file)


def parse_directory(directory_name: dict) -> NormSchedule:
    policy = dict()
    for segment in directory_name['dir'].split("-"):
        match = re.findall(r'EO(\d+)_(\d+)_(\d+)', segment)
        if len(match):
            policy[f'EO{match[0][0]}_start'] = int(match[0][1])
            policy[f'EO{match[0][0]}_duration'] = int(match[0][2])

    return NormSchedule(policy, '2020-06-28')


for policy_directory in directories:
    policy_directory['policy'] = parse_directory(policy_directory)
    target, infected, fitness = evaluator.fitness(policy_directory['absdir'], policy_directory['policy'])
    policy_directory['target'] = -1 * target
    policy_directory['infected'] = infected

# targets = sorted(directories, key=lambda x: x['target'])[:10]

best_score_policy = min(directories, key=lambda x: x['target'])
lowest_infections_policy = min(directories, key=lambda x: x['infected'])
worst_score_policy = max(directories, key=lambda x: x['target'])
highest_infections_policy = max(directories, key=lambda x: x['infected'])

print("\n\n")
print(f"Best score: {best_score_policy['target']:.2e}  had {best_score_policy['infected']} infected ({best_score_policy['infected'] / 119087 * 100:.2f}%)", best_score_policy['absdir'])
print(f"Lowest infections: {lowest_infections_policy['target']:.2e}  had {lowest_infections_policy['infected']} infected ({lowest_infections_policy['infected'] / 119087 * 100:.2f}%)", lowest_infections_policy['absdir'])
print(f"Worst score: {worst_score_policy['target']:.2e}  had {worst_score_policy['infected']} infected ({worst_score_policy['infected'] / 119087 * 100:.2f}%)", worst_score_policy['absdir'])
print(f"Highest infections: {highest_infections_policy['target']:.2e}  had {highest_infections_policy['infected']} infected ({highest_infections_policy['infected'] / 119087 * 100:.2f})%", highest_infections_policy['absdir'])

print("\n\n")
print(lowest_infections_policy['policy'].to_tikz("test-picture/test-picture.tex"))

# print(f"Best score: {best_score_policy['target']:.2e}  had {best_score_policy['infected']} infected ({best_score_policy['infected'] / 119087 * 100:.2f}%)")
# print(f"Lowest infections: {lowest_infections_policy['target']:.2e}  had {lowest_infections_policy['infected']} infected ({lowest_infections_policy['infected'] / 119087 * 100:.2f}%)")
# print(f"Worst score: {worst_score_policy['target']:.2e}  had {worst_score_policy['infected']} infected ({worst_score_policy['infected'] / 119087 * 100:.2f}%)")
# print(f"Highest infections: {highest_infections_policy['target']:.2e}  had {highest_infections_policy['infected']} infected ({highest_infections_policy['infected'] / 119087 * 100:.2f})%")
