import sys

import numpy as np

import utility.utility
from classes.Gyration import Gyration
from utility.run_finder import find_behavior_runs
from utility.utility import *


class MobilityRMSEOnBacklog(object):
    """
    A simple script that just calculates the RMSE for the behavior model on all run outputs present in some
    directory

    This script can be invoked after installing the application with the argument "behavior_rmse", or directly
    from this script, in which case you should pass the county configuration file as the first parameter, and the
    directory to analyse as the second
    """

    def __init__(
        self,
        county_configuration_file: str,
        simulation_output_dir: str,
        va_gyration_mobility_index_file: str = os.path.join(
            "external", "va_county_mobility_index.csv"
        ),
        tick_averages_file_name: str = "tick-averages.csv",
        sliding_window_size: int = 7,
        average_runs: bool = False,
    ):
        toml_file, success = utility.utility.make_file_absolute(
            os.getcwd(), county_configuration_file
        )
        conf = load_toml_configuration(toml_file)
        self.average_runs = average_runs

        self.g = Gyration(
            va_gyration_mobility_index_file,
            tick_averages_file_name,
            conf["counties"],
            sliding_window_size,
        )

        self.runs = find_behavior_runs(simulation_output_dir)
        self.scored_runs = self.score_runs()
        self.print_best_run()

    def score_runs(self):
        scored_runs = dict()
        for key, runs in self.runs.items():
            if self.average_runs:
                scored_runs[key] = np.average(
                    [self.g.calculate_rmse([{0: x}]) for x in runs]
                )
            else:
                scored_runs[key] = self.g.calculate_rmse(
                    [dict(zip(range(len(runs)), runs))]
                )
        return scored_runs

    def print_best_run(self):
        sorted_runs = sorted(self.scored_runs, key=lambda x: self.scored_runs[x])
        best_key = sorted_runs[0]
        score = self.scored_runs[best_key]
        print(
            "{score} was best, with ({0},{1}) for (liberal,conservative), fatigue factor {2} "
            "starting after {3} time steps".format(*best_key, score=score)
        )
        for path in self.runs[best_key]:
            print("\t", path)


if __name__ == "__main__":
    args = sys.argv[1:]
    for index in range(2):
        abspath = make_file_absolute(os.getcwd(), args[index])
        args[index] = abspath[0]
    os.chdir(os.path.join(*[".."] * 2))
    MobilityRMSEOnBacklog(*args)
