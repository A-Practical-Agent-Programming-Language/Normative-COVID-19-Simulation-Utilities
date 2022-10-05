
"""
The new protocol (v3) for a policy that the Bayesian optimization algorithm should output is as follows.
For each norm, a list of possible values is given.
"""
import json

from classes.ExecutiveOrderOptimizer.EOOptimization import EOOptimization
from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule
from classes.ExecutiveOrderOptimizer.policy_v3 import create_policy_specification
from utility.utility import get_project_root



"""
TODO:
    - Calculate penalty based on norm_weights and affected_agents for each value a norm can take (one-week based)
    - Create script to convert norm schedule written in protocol v3 to norm_schedule.csv file
    - Create script to convert protocol v1 to v3
"""


norm_counts = EOOptimization.load_norm_application_counts_from_file(get_project_root(".persistent", "affected-agents-per-norm-65-75-109-540-updated.csv"))
norm_weights = EOOptimization.load_norm_weights(get_project_root("external", "norm_weights_new.csv"))

# TODO Write to JSON file with docs
protocol = create_policy_specification(norm_weights, norm_counts)
for norm, elements in protocol.items():
    print(norm)
    for el in elements:
        print("\t", el)

with open(get_project_root("..", "bayesian-optimization-cost", "policy-protocol-v3.json"), 'w') as json_out:
    json.dump(protocol, json_out, indent=4)

"""
Create a 
"""
ns = NormSchedule.from_norm_schedule("/home/jan/dev/university/2apl/simulation/sim2apl-episimpledemics/src/main/resources/norm-schedule.csv", "2020-06-28")
ns2 = NormSchedule(
    {  # 200 explore, 50 exploit
        'EO0_duration': 2, 'EO0_start': 9, 'EO1_duration': 11, 'EO1_start': 9, 'EO2_duration': 9, 'EO2_start': 2,
        'EO3_duration': 2, 'EO3_start': 9, 'EO4_duration': 5, 'EO4_start': 8, 'EO5_duration': 2, 'EO5_start': 1,
        'EO6_duration': 6, 'EO6_start': 7, 'EO7_duration': 12, 'EO7_start': 17, 'EO8_duration': 11, 'EO8_start': 2},
    "2020-06-28"
)
p3 = ns2.get_protocol_v3()
ns3 = NormSchedule.from_protocol_v3(p3, '2020-06-28')
with open(get_project_root("..", "bayesian-optimization-cost", "example-policy.v3.json"), 'w') as json_out:
    json.dump(p3, json_out, indent=4)

for norm in p3:
    if norm not in protocol:
        print(f"Norm `{norm}` missing from protocol")
for norm in protocol:
    if norm not in p3:
        print(f"Norm `{norm}` missing from schedule")

ns3 = NormSchedule.from_protocol_v3(p3, "2020-06-28")

m1 = ns2.get_norm_event_matrix()
m2 = ns3.get_norm_event_matrix()

assert m1 == m2
