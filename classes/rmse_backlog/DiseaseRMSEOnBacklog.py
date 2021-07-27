import sys

import numpy as np

from classes.Epicurve_RMSE import Epicurve_RMSE
from utility.utility import *


class DiseaseRMSEOnBacklog(object):
    def __init__(
        self,
        county_configuration_file: str,
        simulation_output_dir: str,
        epicurve_filename: str = 'epicurve.sim2apl.csv',
        case_file: str = os.path.join('external', 'va-counties-covid19-cases.csv'),
        minimum_scale: int = 4,
        average_runs: bool = False
    ):
        self.minimum_scale = minimum_scale
        toml_file, success = make_file_absolute(os.getcwd(), county_configuration_file)
        conf = load_toml_configuration(toml_file)
        self.average_runs = average_runs
        self.rmse = Epicurve_RMSE(conf["counties"], epicurve_filename, case_file)

        self.runs = find_runs(
            simulation_output_dir,
            epicurve_filename,
            extract_run_configuration_from_disease_path,
            self.disease_params_dict_to_tuple,
        )

        self.scored_runs = self.score_runs()
        self.print_best_run()

    def score_runs(self):
        scored_runs = dict()
        for key, runs in self.runs.items():
            score: float
            if self.average_runs:
                score = np.average([self.rmse.calculate_rmse(key[2], [{0: x}]) for x in runs])
            else:
                score = self.rmse.calculate_rmse(key[2], [dict(zip(range(len(runs)), runs))])

            if self.minimum_scale is not None and key[2] < self.minimum_scale:
                score = 999999
            scored_runs[key] = np.average(score)

        return scored_runs

    def print_best_run(self):
        sorted_runs = sorted(self.scored_runs, key=lambda x: self.scored_runs[x])
        best_key = sorted_runs[0]
        score = self.scored_runs[best_key]
        print(
            "{score} was best with base infection probabilities ({0}, {1} for symptomatic and asymptomatic (scale {2} "
            "agents respectively (fixed liberal: {3}, conservative: {4}, fatigue factor {5} "
            "starting after {6} time steps)".format(*best_key, score=score)
        )
        for path in self.runs[best_key]:
            print("\t", path)

    @staticmethod
    def disease_params_dict_to_tuple(params_dct: Dict[str, float]) -> tuple:
        key_order = [
            "isymp",
            "iasymp",
            "scale",
            "liberal",
            "conservative",
            "fatigue",
            "fatigue_start",
        ]
        return tuple([params_dct[x] for x in key_order])


if __name__ == "__main__":
    args = sys.argv[1:]
    for index in range(2):
        abspath = make_file_absolute(os.getcwd(), args[index])
        args[index] = abspath[0]
    os.chdir(os.path.join(*[".."] * 2))
    DiseaseRMSEOnBacklog(*args)
