
"""
The new protocol (v3) for a policy that the Bayesian optimization algorithm should output is as follows.
For each norm, a list of possible values is given.
"""
import json

from classes.ExecutiveOrderOptimizer.EOOptimization import EOOptimization
from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule
from utility.utility import get_project_root

protocol = {
    "AllowWearMask": dict(
        values=[False, True]
    ),
    "EncourageSocialDistance": dict(
        values=[False, True]
    ),
    "WearMasInPublicIndoor": dict(
        values=[False, True]
    ),
    "EmployeesWearMask": dict(
        values=[False, True]
    ),
    "MaintainDistance": dict(
        values=[False, True]
    ),
    "EncourageTelework": dict(
        values=[False, True]
    ),
    "SmallGroups": dict(
        values=[False, "100,public", "10,public", "50,public_private", "10,public_private"]
    ),
    "BusinessClosed": dict(
        values=[False, "7 DMV offices", "7 DMV offices;NEB"]
    ),
    "StayHomeSick": dict(
        values=[False, True]
    ),
    "StayHome": dict(
        values=[False, "age>65", "all"]
    ),
    "TakeawayOnly": dict(
        values=[False, True]
    ),
    "ReduceBusinessCapacity": dict(
        values=[False, "50%", "10"]
    ),
    "SchoolsClosed": dict(
        values=[False, "K12", "K12;HIGHER_EDUCATION"]
    )
}

"""
TODO:
    - Calculate penalty based on norm_weights and affected_agents for each value a norm can take (one-week based)
    - Create script to convert norm schedule written in protocol v3 to norm_schedule.csv file
    - Create script to convert protocol v1 to v3
"""


norm_counts = EOOptimization.load_norm_application_counts_from_file(get_project_root(".persistent", "affected-agents-per-norm-65-75-109-540.csv"))
norm_weights = EOOptimization.load_norm_weights(get_project_root("external", "norm_weights.csv"))

"""
Constructs a dictionary with all the possible values each week for a norm can take
"""
for norm in protocol:
    penalties = list()
    for value in protocol[norm]['values']:
        agents = 0
        norm_key = norm
        if value is not True and value:
            norm_key = f'{norm}[{value.replace("public_private", "PP")}]'
        weight = norm_weights[norm_key] if value else 0
        affected = norm_counts[norm_key]["affected_agents"] if value else 0
        penalties.append((weight * affected, weight, affected))

    values = protocol[norm]['values']
    protocol[norm] = list()

    for i, (value, (penalty, weight, affected)) in enumerate(sorted(zip(values, penalties), key=lambda x: x[1][0])):
        protocol[norm].append(dict(value=value, penalty=penalty, threshold=1/len(penalties)*(i+1), weight=weight, affected=affected))

# TODO Write to JSON file with docs
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






# norm -> {
# 	values -> list
# 	weight -> float
# 	affected-agents -> int
# }