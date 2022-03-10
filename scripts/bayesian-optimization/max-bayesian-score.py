import os
import re

from classes.ExecutiveOrderOptimizer.EOOptimization import EOOptimization
from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule
from utility.utility import get_project_root

societal_global_impact_weight = 0.001273665483847564  # used to be: 100/7

agent_population_size = 119087  # TODO what is size? 120k something

norm_counts = dict()
with open(os.path.join(get_project_root(), '../../.persistent', 'affected-agents-per-norm-65-75-109-540.csv'), 'r') as file_in:
    file_in.readline()  # Skip header
    for line in file_in:
        match = re.findall(r'(\w+(?:\[[\w,;>%]+])?);(\d+);(\d+)', line)
        if match:
            norm_counts[match[0][0]] = {'affected_agents': int(match[0][1]),
                                        'affected_duration': int(match[0][2])}

norm_weight_files = ['norm_weights_favour_economy.csv', 'norm_weights_favour_schools_open.csv', 'norm_weights.csv']
norm_weights = dict(zip(norm_weight_files, map(
    lambda x: EOOptimization.load_norm_weights(os.path.join(get_project_root(), '../../external', x)),
    norm_weight_files
)))


eos = dict()
for i in range(9):
    eos[f"EO{i}_start"] = 0
    eos[f"EO{i}_duration"] = 17
max_norm_schedule = NormSchedule(eos, "2020-06-28")

for i in range(9):
    eos[f"EO{i}_start"] = 17
    eos[f"EO{i}_duration"] = 1
nin_norm_schedule = NormSchedule(eos, "2020-06-28")

for norm_schedule, name, infected in [(max_norm_schedule, "theoretical upper bound", agent_population_size)]:  #, (nin_norm_schedule, "theoretical lower bound", 0)]:
    for weight_file, weights in norm_weights.items():
        fitness = 0
        for norm in norm_counts.keys():
            active_duration = norm_schedule.get_active_duration(norm)
            affected_agents = .6 * infected if norm == "StayHomeSick" else norm_counts[norm]["affected_agents"]
            # affected_agents = norm_counts[norm]["affected_agents"]
            norm_weight = weights[norm]
            fitness += (active_duration * norm_weight * affected_agents)
        target = (infected + (societal_global_impact_weight * fitness))

        print(name, weight_file, f"{target:.2e}")

# Assuming everybody gets infected:
    # Assuming default weights, and using counts as they were:
    # Target = 184205965.0
    # If we do not use special case of stay home if sick, it becomes: 147765955. That is quite a significant decrease!?

    #assuming favour schools open:
        # target = 73942520.0

    # Favour economy:
        # 203145580.00000003

# Assuming everybody gets infected:
    # Assuming default weights, and using counts as they were:

    # If we do not use special case of stay home if sick, it becomes: 147765955. That is quite a significant decrease!?
# 184205965.0 (default)
# 73942520.0 (schools)
# 203145580.00000003 (economy)
# 19634824 (actual target)


