import os
from collections import defaultdict
from typing import List, Dict

import numpy as np

from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule


class EOEvaluator(object):

    def __init__(
            self,
            societal_global_impact_weight: float,
            norm_weights: Dict[str, float],
            norm_counts: Dict[str, Dict[str, int]]
    ):
        self.societal_global_impact_weight = societal_global_impact_weight
        self.norm_weights = norm_weights
        self.norm_counts = norm_counts

    def fitness(self, directories: List[Dict[int, str]], norm_schedule: NormSchedule) -> (float, int, float):
        fitness = 0
        for norm in self.norm_counts.keys():
            active_duration = norm_schedule.get_active_duration(norm)
            affected_agents = self.find_number_of_agents_affected_by_norm(norm, directories)
            norm_weight = self.norm_weights[norm]
            fitness += active_duration * norm_weight * affected_agents
        infected = self.count_infected_agents(directories)
        final_fitness = self.societal_global_impact_weight * fitness
        return -1 * (infected + final_fitness), infected, fitness

    def find_number_of_agents_affected_by_norm(self, norm_name: str, directories: List[Dict[int, str]]) -> int:
        if "StayHomeSick" in norm_name:
            return self.find_number_of_agents_affected_by_stayhome_if_sick(directories)
        else:
            return self.norm_counts[norm_name]['affected_agents']

    def find_number_of_agents_affected_by_stayhome_if_sick(self, directories: List[Dict[int, str]]) -> int:
        # TODO, just report number of symptomatically ill people
        # TODO How to find people who were symptomatic?
        return round(.6 * self.count_infected_agents(directories))

    @staticmethod
    def count_infected_agents(directories: List[Dict[int, str]]) -> int:
        amounts = defaultdict(int)
        for node in directories:
            for run, directory in node.items():
                with open(os.path.join(directory, 'epicurve.pansim.csv'), 'r') as file_in:
                    headers = file_in.readline()[:-1].split(",")
                    values = file_in.readlines()[-1][:-1].split(",")
                    infected = sum(map(lambda x: int(values[headers.index(x)]), [
                        "expo",
                        "isymp", "iasymp", "recov"]))
                    amounts[run] += infected
        return round(np.average(list(amounts.values())))
