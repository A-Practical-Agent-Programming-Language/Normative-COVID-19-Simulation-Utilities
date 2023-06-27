import json
import os
from collections import defaultdict
from typing import List, Dict

import numpy as np

from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule


class EOEvaluator(object):

    def __init__(self, societal_global_impact_weight: float, dry_run=False, **kwargs):
        self.dry_run = dry_run
        self.societal_global_impact_weight = societal_global_impact_weight
        if 'norm_weights' in kwargs and 'norm_counts' in kwargs:
            self.__init_from_weights_and_counts(**kwargs)
        elif 'policy_file' in kwargs:
            self.__init_from_policy_file(**kwargs)
        elif 'policy_specification' in kwargs:
            self.__init_from_policy_object(**kwargs)
        else:
            raise ValueError("Expecting either a policy specification file, or norm weights and norm counts files")

    def __init_from_policy_file(self, policy_file: str):
        with open(policy_file, 'r') as f_obj:
            json_object = json.load(f_obj)
            self.__init_from_policy_object(json_object)

    def __init_from_policy_object(self, policy_specification):
        self.norm_weights: Dict[str, float] = dict()
        self.norm_counts: Dict[str, Dict[str, int]] = dict()

        items = list(policy_specification['static'].items()) + (list(policy_specification["dynamic"].items()))

        for norm, values in items:
            if type(values[1]['value']) == bool and values[1]['value']:
                self.norm_weights[norm] = values[1]['weight']
                self.norm_counts[norm] = dict(
                    affected_duration=values[1]['seconds_affected'],
                )
            else:
                for value in values[1:]:
                    norm_key = f"{norm}[{value['value']}]"
                    self.norm_weights[norm_key] = value['weight']
                    self.norm_counts[norm_key] = dict(
                        affected_duration=value['seconds_affected'],
                    )

    def __init_from_weights_and_counts(
            self,
            norm_weights: Dict[str, float],
            norm_counts: Dict[str, Dict[str, int]]
    ):
        self.norm_weights = norm_weights
        self.norm_counts = norm_counts

    def fitness(self, directories: List[Dict[int, str]], norm_schedule: NormSchedule) -> (float, int, float):
        infected, n_agents = self.count_infected_agents(directories, self.dry_run)
        return self.fitness_from_values(infected, n_agents, norm_schedule)

    def fitness_from_values(self, infected_agents: int, n_agents: int, norm_schedule: NormSchedule) -> (
            float, int, float):
        fitness = 0
        for norm in [x for x, y in self.norm_counts.items() if
                     x != "SmallGroups[250,PP]" and (y["affected_duration"] > 0)]:
            active_duration = norm_schedule.get_active_days(norm) / 7
            affected_agents = self.find_number_of_agents_affected_by_norm(norm, infected_agents, n_agents)
            norm_weight = self.norm_weights[norm]
            penalty = active_duration * norm_weight * affected_agents
            fitness += penalty

        final_fitness = self.societal_global_impact_weight * fitness
        return (infected_agents + final_fitness), infected_agents, fitness

    def find_number_of_agents_affected_by_norm(self, norm_name: str, infected_agents, total_agents: int) -> float:
        """
        Given a norm, returns the number of agents affected by that norm, weighted by the duration of those activities
        In all cases but one, this is determined statically based on the activity schedules of the synthetic population.
        In the case of StayHomeWhenSick, this number depends on the total number of agents who have been symptomatically
        ill
        Args:
            norm_name:
            infected_agents:
            total_agents:

        Returns:

        """
        if "StayHomeSick" in norm_name:
            return self.find_number_of_agents_affected_by_stayhome_if_sick(infected_agents, total_agents)
        else:
            return self.norm_counts[norm_name]['affected_duration']

    def find_number_of_agents_affected_by_stayhome_if_sick(self, infected_agents: int, total_agents: int) -> float:
        # The stored value is the total duration of activities during one week, divided by the total number of agents
        # (to get the average duration _per agent_), multiplied by 0.6 (which reflects the percentage of symptomatic
        # infections compared to the overall number of infections). The real duration of affected activities, then,
        # should be the number of infected agents multiplied by this number we have found with the above.
        return self.norm_counts["StayHomeSick"]["affected_duration"] * infected_agents / total_agents * .6
        # return round(self.norm_counts["StayHomeSick"]["affected_duration"] * 0.6 * (infected_agents / total_agents))
        # return self.norm_counts["StayHomeSick"]["affected_duration"] * infected_agents + 1.0

    @staticmethod
    def count_infected_agents(directories: List[Dict[int, str]], dry_run: bool = False) -> (int, int):
        total_infected_agents = defaultdict(int)
        total_population_size = defaultdict(int)
        for node in directories:
            for run, directory in node.items():
                with open(os.path.join(directory, f'epicurve.{"sim2apl" if dry_run else "pansim"}.csv'),
                          'r') as file_in:
                    headers = file_in.readline()[:-1].split(";" if dry_run else ",")
                    values = file_in.readlines()[-1][:-1].split(";" if dry_run else ",")
                    if dry_run:
                        relevant_headers = ["EXPOSED", "INFECTED_SYMPTOMATIC", "INFECTED_ASYMPTOMATIC", "RECOVERED"]
                        other_pop_headers = ["NOT_SET", "SUSCEPTIBLE"]
                    else:
                        relevant_headers = ["expo", "isymp", "iasymp", "recov"]
                        other_pop_headers = ["succ"]
                    infected = sum(map(lambda x: int(values[headers.index(x)]), relevant_headers))
                    total_infected_agents[run] += infected
                    total_population_size[run] += sum(
                        map(lambda x: int(values[headers.index(x)]), relevant_headers + other_pop_headers))
        return (
            round(np.average(list(total_infected_agents.values()))),
            round(np.average(list(total_population_size.values())))
        )
