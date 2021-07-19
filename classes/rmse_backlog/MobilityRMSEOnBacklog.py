import sys

import utility.utility
from classes.Gyration import Gyration
from utility.utility import *


class MobilityRMSEOnBacklog(object):
    """
    A simple script that just calculates the RMSE for the behavior model on all run outputs present in some
    directory
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
    ):
        toml_file, success = utility.utility.make_file_absolute(
            os.getcwd(), county_configuration_file
        )
        conf = load_toml_configuration(toml_file)

        self.g = Gyration(
            va_gyration_mobility_index_file,
            tick_averages_file_name,
            conf["counties"],
            sliding_window_size,
        )

        self.runs = find_runs(
            simulation_output_dir,
            tick_averages_file_name,
            extract_run_configuration_from_behavior_path,
            self.behavior_params_dict_to_tuple,
        )
        self.scored_runs = self.score_runs()
        self.print_best_run()

    def score_runs(self):
        scored_runs = dict()
        for key, runs in self.runs.items():
            scored_runs[key] = self.g.calculate_rmse(runs)
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

    @staticmethod
    def behavior_params_dict_to_tuple(params_dct: Dict[str, float]) -> tuple:
        return (
            params_dct["liberal"],
            params_dct["conservative"],
            params_dct["fatigue"],
            params_dct["fatigue_start"],
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    for index in range(2):
        abspath = make_file_absolute(os.getcwd(), args[index])
        args[index] = abspath[0]
    os.chdir(os.path.join(*[".."]*2))
    MobilityRMSEOnBacklog(*args)
