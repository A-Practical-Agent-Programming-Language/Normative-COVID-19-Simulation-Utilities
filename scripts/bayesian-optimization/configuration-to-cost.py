"""

This is a quick Python file that shows how the cost of a policy is calculated and how, from that, the fitness
of a policy is derived.

During the optimization process, this is calculated by :func:`Classes.ExecutiveOrderOptimzer.EOEvaluator.fitness`, which
also reads the outputs of the simulation runs to determine the number of infections the policy results in.

The cost of a policy is defined as the sum of the duration multiplied by the number of agents affected multiplied by
the relative weight of each norm in the policy.

The fitness is the negative of the number of infected agents and the cost multiplied by the global weight omega.

Note: The entire calibration process needs to be installed. In the root directory of this repository:
    $ pip install -U -e .

"""
import sys
from typing import List, Dict

import numpy as np

from classes.ExecutiveOrderOptimizer.EOOptimization import EOOptimization
from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule
from utility.utility import get_project_root

Policy = Dict[str, int or float]


def configuration_to_policy(policy: Policy) -> NormSchedule:
    """
    We convert a configuration of parameters to a norm schedule in three steps.
    First, all values are rounded to the nearest integer, then, all the norms are added to an event schedule
    based on when the EOs to which they belong are active, and lastly, conflicts are resolved, meaning some norms
    can be temporarily deactivated
    Args:
        policy: Parameter configuration of a policy

    Returns:
        A NormSchedule object containing the actual policy as used for simulation
    """
    policy = EOOptimization.normalize_params(policy)
    return NormSchedule(policy, "2020-06-28")


def cost(
        policy: Policy,
        infected: List[int],
        norm_weights_file: str,
        affected_agents_file: str
) -> float:
    """
    This method calculates the cost of a policy. It returns a real-valued positive number that is the sum of the duration
    multiplied by the impact of each norm.

    There is one norm, StayHomeIfSick, that uses the number of symptomatically infected agents in the simulation,
    so the cost can vary slightly between runs. The number of symptomatically infected agents is assumed to be 60%
    of the total number of infected agents in a simulation, due to the probability to go to the infected state from
    the exposed state of .6

    Args:
        policy:                 Dictionary of starting points and durations for each EO in the policy
        infected:               A list of the number of infected agents in each simulation run for this policy
        norm_weights_file:      The file in which all norms in the policy are assigned a (relative) weight
        affected_agents_file:   The file that contains the numbers of agents affected by each norm in the policy

    Returns:
        The cost of the policy
    """
    norm_weights, affected_agents = load_files(norm_weights_file, affected_agents_file)
    ns = configuration_to_policy(policy)
    policy_cost = 0
    for norm in affected_agents:
        active_duration = ns.get_active_duration(norm)
        agents = .6 * round(np.average(infected)) if "StayHomeSick" in norm else affected_agents[norm]['affected_agents']
        weight = norm_weights[norm]

        policy_cost += active_duration * weight * agents

    return policy_cost


def fitness(
        policy: Policy,
        infected: List[int],
        omega: float,
        norm_weights_file: str,
        affected_agents_file: str
) -> float:
    """
    The fitness is the number of infections plus the weighted :func:`cost` of the policy, multiplied by -1 because
    we are maximizing.

    Args:
        policy:                 Dictionary of starting points and durations for each EO in the policy
        infected:               A list of the number of infected agents in each simulation run for this policy
        omega:                  The weight that determines the importance of the policy cost vs. number of infections
        norm_weights_file:      The file in which all norms in the policy are assigned a (relative) weight
        affected_agents_file:   The file that contains the numbers of agents affected by each norm in the policy

    Returns: Negative real-valued fitness of the policy
    """
    actual_cost = cost(policy, infected, norm_weights_file, affected_agents_file)
    return -1 * (round(np.average(infected)) + (actual_cost * omega))


def load_files(norm_weights_file: str, affected_agents_file: str):
    norm_weights = EOOptimization.load_norm_weights(norm_weights_file)
    affected_agents = EOOptimization.load_norm_application_counts_from_file(affected_agents_file)
    return norm_weights, affected_agents


if __name__ == "__main__":
    """
    A policy maker ranks the desirability of interventions by assigning a weight. This allows the policy maker to make
    the cost of one intervention relatively higher or lower than others.

    We currently use three different models for this
    """
    f_norm_weights = dict(
        default=get_project_root("external", "norm_weights.csv"),
        schools=get_project_root("external", "norm_weights_favour_schools_open.csv"),
        economy=get_project_root("external", "norm_weights_favour_economy.csv")
    )

    """
    The number of affected agents is calculated with Sim-2APL, and stored in the following file
    """
    f_affected_agents = get_project_root(".persistent", "affected-agents-per-norm-65-75-109-540.csv")

    """
    Omega (the global norm weight) determines the relative impact of the cost of a policy vs. the outcome
        (i.e., the number of infected agents).

        This value is set such that in the actually implemented policy, the policy cost and number of infections contribute
        equally to the fitness.
    """
    omega = 0.0022626371521808136

    policy_to_evaluate: Policy = {
        "EO0_duration": 6.9926419015578,
        "EO0_start": 16.162143208968576,
        "EO1_duration": 12.711903068982481,
        "EO1_start": 10.177194231349622,
        "EO2_duration": 3.4962982470789843,
        "EO2_start": 2.651906845715445,
        "EO3_duration": 1.9293377946911914,
        "EO3_start": 14.724994478173898,
        "EO4_duration": 10.61784018789134,
        "EO4_start": 12.037233822532773,
        "EO5_duration": 1.3293519087328391,
        "EO5_start": 16.488467486753905,
        "EO6_duration": 14.319082252806748,
        "EO6_start": 3.609764881530695,
        "EO7_duration": 3.90919947531361,
        "EO7_start": 3.1178766675083747,
        "EO8_duration": 5.867875887352604,
        "EO8_start": 8.920859337748043
    }

    # 5 simulation runs result in 5 different number of infected agents at the end of each simulation
    infected_per_run = [64241, 64583, 64044, 64880, 63751]

    """
    The average number of infected agents is normally extracted from the simulation output using
    :func:`classes.ExecutiveOrderOptimizer.EOOptimization.EOEvaluator.count_infected_agents`
    """
    average_infected = round(np.average(infected_per_run))

    print(f"Assuming {average_infected} infected agents on average:")
    print(
        f"\tPolicy cost:",
        cost(policy_to_evaluate, infected_per_run, f_norm_weights['default'], f_affected_agents)
    )
    print(
        "\tPolicy fitness:",
        fitness(policy_to_evaluate, infected_per_run, omega, f_norm_weights['default'], f_affected_agents)
    )
