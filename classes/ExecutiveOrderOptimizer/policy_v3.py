import json
import math
import os
import sys
from pathlib import Path
from typing import List, Dict

from scipy.stats.qmc import Sobol

####################################################################################################################
#
#
# These two methods are what I would like your confirmation on
#
#
####################################################################################################################
from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule


def generate_random_policy(n_weeks: int = 17, init_evals: int = 1):
    """
    Generates a matrix of 13xn_weeks with random values between 0 and 1.
    This is the raw output of the optimization algorithm, that we need to convert
    to a policy matrix
    """
    assert math.sqrt(init_evals) % 1 == 0, f"Value of init_evals should be a power of two. Got {init_evals}"

    qmc_gen = Sobol(d=13 * n_weeks, scramble=True, seed=1)
    param_list = qmc_gen.random(init_evals)

    policy_matrices = [
        [list(single_policy_param_list[i * n_weeks:i * n_weeks + n_weeks]) for i in range(13)]
        for single_policy_param_list in param_list
    ]

    return policy_matrices


def params_to_policy_v3(static_params: List[List[float]], dynamic_param: List[float]):
    """
    Converts a list of 13 parameter values to a policy matrix, following Policy version 3
    Args:
        static_params: List of 13 parameters, each in the range (0,1)

    Returns:
        A matrix, i.e., dictionary with norms as key, and a list with norm parameters as values,
        in which each item indicates the status of that norm for a specific week
    """
    assert len(static_params) == 12, ValueError("13 lists required (1 list for each norm)")
    assert \
        all(map(lambda week_list: len(week_list) == len(static_params[0]), static_params + [dynamic_param])), \
        ValueError("The list corresponding to the values for each week need to be equal length for all 13 norms")
    assert \
        all(map(lambda week_list: all(map(lambda param: 0 <= param <= 1, week_list)), static_params + [dynamic_param])), \
        ValueError("All parameters must be between 0 and 1")

    norm_matrix = dict()

    for norm, static_params in zip(ordered_norms_list['static'], static_params):
        norm_matrix[norm] = list()
        for param in static_params:
            for item in policy_specification['static'][norm]:
                if item['threshold'] >= param:
                    norm_matrix[norm].append(item['value'])
                    break
    norm_matrix['StayHomeSick'] = list()
    for param in dynamic_param:
        for item in policy_specification['dynamic']['StayHomeSick']:
            if item['threshold'] >= param:
                norm_matrix['StayHomeSick'].append(item['value'])

    return norm_matrix


####################################################################################################################
#
#
# Example of how to score policy based on a V3 policy
#
#
####################################################################################################################

def score_from_policy_v3(policy_params: Dict[str, List], infected, weights_file: str = 'weights_default_new.json') -> (
        float, float):
    """
    Calculate the policy fitness from a Policy V3 Matrix.
    Args:
        policy_params: The Policy V3 Matrix
        infected: The number of agents infected during the simulation using these policy_params

    Returns:
        A tuple,
            First element: Overall policy fitness
            Second element: The sum of the penalties of all norms in the policy V3 Matrix
    """
    with open(weights_file, 'r') as weights_new_in:
        """This is one of the files I sent you, and also the output of create_policy_specification() below"""
        weights = json.load(weights_new_in)

    sum_penalties = 0
    for intervention, intervention_params in policy_params.items():
        category = "dynamic" if "StayHomeSick" == intervention else "static"
        possible_params = weights[category][intervention]
        for intervention_param in intervention_params:
            param = next(filter(lambda p: p["value"] == intervention_param, possible_params))
            if intervention == "StayHomeSick":
                penalty = round(param['seconds_affected'] * .6 * (infected / 119087))
            else:
                penalty = param["seconds_affected"]
            sum_penalties += (7 * param['weight'] * penalty)
        break
    weighted_penalty = 1.5965549750773233e-10 * sum_penalties

    return -1 * (infected + weighted_penalty), sum_penalties


####################################################################################################################
#
#
# THE FOLLOWING IS JUST TO MAKE SURE THIS SCRIPT HAS ALL THE REQUIRED DATA AVAILABLE. NOT NECESSARY TO CHECK
#
#
####################################################################################################################

"""
The protocol is a dictionary that specifies all the available norms, and for
each norm specifies the possible parameters. A value of False is a required parameter
value possibility indicating the norm (no version of it) is active.
For norms without parameters, a value of True indicates it is active
"""
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
        values=[False, "100,public", "10,public", "50,PP", "10,PP"]
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
If we want to deterministically convert a list of parameters to a policy matrix, the index of each parameter in the
list needs to be unambiguously associated with a norm. We use alphabetic ordering to ensure this.
"""
static_norm_names = [p for p in protocol if p != "StayHomeSick"]
ordered_norms_list = dict(
    static=sorted(list(static_norm_names)),
    dynamic=["StayHomeSick"]
)


def create_policy_specification(norm_weights, norm_counts):
    """
    Constructs a dictionary with all the possible values each week for a norm can take
    """
    actual_protocol = dict(static=dict(), dynamic=dict())
    for norm in protocol:
        penalties = list()
        for value in protocol[norm]['values']:
            norm_key = norm
            if value is not True and value:
                norm_key = f'{norm}[{value}]'
            weight = norm_weights[norm_key] if value else 0
            seconds_affected = norm_counts[norm_key]["total_duration"] if value else 0

            multiplier = 1
            if norm == "StayHomeSick":
                # Here we are going to do some weird stuff:
                # For each norm, normally the penalty is active_duration * n_affected * norm_weight
                # For stay home sick, however, the active duration depends on the actual number of infected agents.
                # The value we have (at norm_weights[norm_key]["total_duration"]) is the total number of seconds of affected
                # activities for a single week, assuming EVERY agent gets infected. What we need to do then, is divide this
                # number by the total population size, which is given by the "affected_agents" key. Moreover, only symptomatically
                # ill agents will know the norm applies to them, which, on average, is about 60% of infected agents.
                # For this reason, we then *multiply* again with .6.
                # So contrary to all other values, this value is per infected agent, instead of total (but this is a
                # special case already, which is why we put it in the dynamic part of the policy specification)
                n_total_agents = norm_counts[norm_key]["affected_agents"]
                multiplier = .6 / n_total_agents

            penalties.append((weight * seconds_affected * multiplier, weight, seconds_affected))

        values = protocol[norm]['values']
        key = 'dynamic' if norm == "StayHomeSick" else 'static'
        actual_protocol[key][norm] = list()

        for i, (value, (penalty, weight, seconds_affected)) in enumerate(
                sorted(zip(values, penalties), key=lambda x: x[1][0])):
            actual_protocol[key][norm].append(
                dict(value=value, penalty=penalty, threshold=1 / len(penalties) * (i + 1), weight=weight,
                     seconds_affected=seconds_affected))

    return actual_protocol


def load_norm_weights(norm_weights_file: str) -> Dict[str, float]:
    """
    Reads the specified norm weights file as a dictionary
    Args:
        norm_weights_file: File location specifying relative norm weights

    Returns:

    """
    norm_weights = dict()
    with open(norm_weights_file, 'r') as norm_weights_in:
        norm_weights_in.readline()  # Skip header
        for line in norm_weights_in:
            data = line.split("\t")
            norm_weights[data[0]] = float(data[1])
    return norm_weights


def load_affected_agents(affected_agents_file: str) -> Dict[str, Dict[str, int]]:
    norm_counts = dict()
    with open(affected_agents_file, 'r') as file_in:
        file_in.readline()  # Skip header
        for line in file_in:
            data = line.split("\t")
            norm_counts[data[0]] = {'affected_agents': int(data[1]), 'affected_duration': int(data[2]),
                                    'total_duration': int(data[2])}

    return norm_counts


def populate_norm_schedules(total, directories):
    policies = generate_random_policy(init_evals=total)  # Should be a power of two
    for (i, params) in enumerate(policies):
        out_f = os.path.join(directories[i % len(directories)], "{0}", f'exploration-policy-{i}.{{1}}')
        csv_out = out_f.format("norm-schedule", "csv")
        json_out = out_f.format("protocol", "json")
        policy_v3 = params_to_policy_v3(params[:-1], params[-1])
        ns = NormSchedule.from_protocol_v3(policy_v3, last_simulation_date='2020-06-28', is_weeks=True)
        ns.write_to_file(csv_out)
        os.makedirs(Path(json_out).parent, exist_ok=True)
        with open(json_out, 'w') as json_out_descriptor:
            json.dump(policy_v3, json_out_descriptor, indent=1)
        ns2 = NormSchedule.from_norm_schedule(csv_out, '2020-06-28')
        protocol_sanity_check, rounding_error = ns2.get_protocol_v3()
        assert not rounding_error, "Got a rounding error trying to obtain the policy?"
        assert policy_v3 == protocol_sanity_check, "Written policy is not the same as origin"
        print(f"Created {csv_out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Please specify the file 'norm_weights.csv' as the first argument and the file "
            "'affected-agents-per-norm-65-75-109-540.csv' as the second"
        )
        exit(0)

    # Load the assigned norm weights, and the counted number of agents, so we can determine threshold values for
    # norm parameters
    norm_weights_map = load_norm_weights(sys.argv[1])
    norm_counts_map = load_affected_agents(sys.argv[2])

    # Create the policy protocol
    policy_specification = create_policy_specification(norm_weights_map, norm_counts_map)
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'w') as policy_specification_out:
            json.dump(policy_specification, policy_specification_out, indent="\t")

    # # Generate the random parameters for a policy
    # random_policy_params = generate_random_policy(init_evals=1)
    #
    # # Generate a random policy for 14 weeks
    # policy = random_policy_params[0]  # We requested just one policy, so list should have just 1 element
    # static_params = policy[:-1]
    # dynamic_param = policy[-1]
    # policy = params_to_policy_v3(static_params, dynamic_param)
    #
    # populate_norm_schedules(256, [get_project_root("initial_norm_schedules", f"run-{i}") for i in range(10)])
