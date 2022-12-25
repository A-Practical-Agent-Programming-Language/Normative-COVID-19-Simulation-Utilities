import json
import os
from collections import defaultdict
from typing import List, Dict

import numpy as np

from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule


class EOEvaluator(object):

    def __init__(self, societal_global_impact_weight: float, **kwargs):
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
        for norm, values in policy_specification['static'].items():
            if type(values[0]['value']) == bool and not values[0]['value']:
                self.norm_weights[norm] = values[1]['weight']
                self.norm_counts[norm] = dict(
                    affected_duration=values[1]['seconds_affected'],
                    total_duration=values[1]['seconds_affected']
                )
            else:
                for value in values[1:]:
                    norm_key = f"{norm}[{value['value']}"
                    self.norm_weights[norm_key] = value['weight']
                    self.norm_counts[norm_key] = dict(
                        affected_duration=value['seconds_affected'],
                        total_duration=value['seconds_affected']
                    )

    def __init_from_weights_and_counts(
            self,
            norm_weights: Dict[str, float],
            norm_counts: Dict[str, Dict[str, int]]
    ):
        self.norm_weights = norm_weights
        self.norm_counts = norm_counts

    def fitness(self, directories: List[Dict[int, str]], norm_schedule: NormSchedule) -> (float, int, float):
        fitness = 0
        for norm in [x for x, y in self.norm_counts.items() if x != "SmallGroups[250,PP]" and (y["affected_duration"] > 0)]:
            active_duration = norm_schedule.get_active_duration(norm) / 7
            affected_agents = self.find_number_of_agents_affected_by_norm(norm, directories)
            norm_weight = self.norm_weights[norm]
            fitness += active_duration * norm_weight * affected_agents
        infected, n_agents = self.count_infected_agents(directories)
        final_fitness = self.societal_global_impact_weight * fitness
        return (infected + final_fitness), infected, fitness

    def find_number_of_agents_affected_by_norm(self, norm_name: str, directories: List[Dict[int, str]]) -> int:
        if "StayHomeSick" in norm_name:
            return self.find_number_of_agents_affected_by_stayhome_if_sick(directories)
        else:
            return self.norm_counts[norm_name]['affected_duration']

    def find_number_of_agents_affected_by_stayhome_if_sick(self, directories: List[Dict[int, str]]) -> int:
        # TODO, just report number of symptomatically ill people
        # TODO How to find people who were symptomatic?
        infected, n_agents = self.count_infected_agents(directories)
        return round(self.norm_counts["StayHomeSick"]["affected_duration"] * 0.6 * (infected / n_agents))

    @staticmethod
    def count_infected_agents(directories: List[Dict[int, str]]) -> (int, int):
        total_infected_agents = defaultdict(int)
        total_population_size = defaultdict(int)
        for node in directories:
            for run, directory in node.items():
                with open(os.path.join(directory, 'epicurve.pansim.csv'), 'r') as file_in:
                    headers = file_in.readline()[:-1].split(",")
                    values = file_in.readlines()[-1][:-1].split(",")
                    infected = sum(map(lambda x: int(values[headers.index(x)]), [
                        "expo",
                        "isymp", "iasymp", "recov"]))
                    total_infected_agents[run] += infected
                    total_population_size[run] += sum(map(int, values))
        return (
            round(np.average(list(total_infected_agents.values()))),
            round(np.average(list(total_population_size.values())))
        )
