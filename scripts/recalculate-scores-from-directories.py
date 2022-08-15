import os
import re
from collections import defaultdict
from typing import Dict

import click

from os.path import join

from classes.Gyration import Gyration
from utility.utility import get_project_root, load_toml_configuration


class ScoreRecalculator(object):

    def __init__(self, simulation_output, config):
        self.simulation_output = simulation_output
        self.config = config
        self.recalculate()

    def recalculate(self):
        for subdir in os.listdir(self.simulation_output):
            grouped_runs = self.group_runs(subdir)
            if "disease" in subdir:
                self.recalculate_disease(grouped_runs)
            elif "behavior" in subdir:
                self.recalculate_behavior(grouped_runs)

    def recalculate_behavior(self, grouped_behavior_runs):
        t = load_toml_configuration(self.config)
        g = Gyration(
            get_project_root("external", "va_county_mobility_index.csv"),
            "tick-averages.csv",
            t["counties"],
            7
        )
        scores = defaultdict(dict)
        for simulation, directories in grouped_behavior_runs.items():
            run_scores = 0
            for directory in directories.values():
                run_scores += g.calculate_rmse([{0 : directory}])
            scores[simulation]["old"] = run_scores / len(directories)
            scores[simulation]["new"] = g.calculate_rmse([directories])

        best_scores = sorted(list(scores.keys()), key=lambda x: scores[x]["old"])
        for score in best_scores:
            print(score, scores[score])

    def recalculate_disease(self, grouped_disease_runs):
        pass

    def group_runs(self, runs_path) -> Dict[str, Dict[int, str]]:
        simulation_outputs = defaultdict(dict)
        for directory in os.listdir(join(self.simulation_output, runs_path)):
            print(directory)
            for simulation_output in os.listdir(join(self.simulation_output, runs_path, directory)):
                simulation, run = self.is_simulation(runs_path, directory, simulation_output)
                if simulation:
                    simulation_outputs[simulation][run] = join(self.simulation_output, runs_path, directory, simulation_output)

        return simulation_outputs

    def is_simulation(self, *directories):
        if "tick-averages.csv" in os.listdir(join(self.simulation_output, *directories)):
            match = re.findall('^(.*)-run(\d+)', directories[-1])
            if match:
                return match[0][0], int(match[0][1])
        return None, -1


@click.command()
@click.option("-s", "--simulation-output", help="All simulation runs", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("-c", "--config", help="county configuration", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def recalculate_scores(**args):
    ScoreRecalculator(**args)


if __name__ == "__main__":
    recalculate_scores()
