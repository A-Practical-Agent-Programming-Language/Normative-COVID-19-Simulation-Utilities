import datetime
import os.path
import re
import subprocess
from collections import defaultdict
from typing import List, Dict

import bayes_opt.util
import numpy as np

from classes.ExecutiveOrderOptimizer import NormService
from .NormService import date_from_data_point
from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule
from classes.execution.CodeExecution import CodeExecution
from utility.utility import get_project_root

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events


class EOOptimization(CodeExecution):
    rundirectory_template = [
        "optimization",
        "{ncounties}counties-fips-{fips}",
        "{liberal}l-{conservative}c-{fatigue}f-{fatigue_start}fs-run{run}-EO0_{x[EO0_start]}_{x[EO0_duration]}-"
        "EO1_{x[EO1_start]}_{x[EO1_duration]}-EO2_{x[EO2_start]}_{x[EO2_duration]}-"
        "EO3_{x[EO3_start]}_{x[EO3_duration]}-EO4_{x[EO4_start]}_{x[EO4_duration]}-"
        "EO5_{x[EO5_start]}_{x[EO5_duration]}-EO6_{x[EO6_start]}_{x[EO6_duration]}-"
        "EO7_{x[EO7_start]}_{x[EO7_duration]}-EO8_{x[EO8_start]}_{x[EO8_duration]}",
    ]
    progress_format = "[OPTIMIZATION] [{time}] {ncounties} counties ({fips}): {score} for x={x} policy optimization (dir={output_dir})\n"
    csv_log = os.path.join(get_project_root(), "output", "optimization.results.csv")
    json_log = os.path.join(get_project_root(), "output", "optimization-logs.json")

    def __init__(
            self,
            societal_global_impact_weight: float,
            norm_weights: str,
            alpha=0.5,
            mode_liberal=0.5,
            mode_conservative=0.5,
            fatigue=0.0125,
            fatigue_start=60,
            *args,
            **kwargs
    ):
        super(EOOptimization, self).__init__(*args, **kwargs)
        self.mode_liberal, self.mode_conservative = mode_liberal, mode_conservative
        self.fatigue, self.fatigue_start = fatigue, fatigue_start
        self.run_configuration["liberal"] = self.mode_liberal
        self.run_configuration["conservative"] = self.mode_conservative
        self.run_configuration["fatigue"] = self.fatigue
        self.run_configuration["fatigue_start"] = self.fatigue_start

        self.alpha = alpha
        self.societal_global_impact_weight = societal_global_impact_weight
        self.norm_weights_file = norm_weights
        self.norm_counts = self.load_norm_application_counts()
        self.norm_weights = self.load_norm_weights()

        self.start_optimization()

    def simple_test_f(self, **x):
        """
        Just a simple method that converts the 20 values x can take to a deterministic value using a polynomial function
        """
        val = 0
        x = self.normalize_params(x)
        keys = list(x.keys())
        for i in range(len(keys)):
            val += (i+1) * int(round(x[keys[i]]))
        return -1 * val

    def normalize_params(self, x):
        new_x = dict()
        for key, val in x.items():
            new_x[key] = int(round(val))
        return new_x

    def calibrate(self, **x):
        print(x)
        return super(EOOptimization, self).calibrate(x)

    def start_optimization(self):
        optimizer = BayesianOptimization(
            f=self.calibrate,  # self.simple_test_f,  # flip around for quick test
            pbounds=NormService.get_bounds(),
            random_state=42,
            verbose=1
        )

        if os.path.exists(self.json_log):
            bayes_opt.util.load_logs(optimizer, [self.json_log])

        # From documentation:
        #   "By default the previous data in the json file is removed. If you want to keep working with the same logger,
        #   the reset paremeter in JSONLogger should be set to False."
        # However, init does not take that parameter (yet?), so we do a workaround
        logger = JSONLogger(path=os.path.join(get_project_root(), ".persistent", ".tmp", "tmp_bayesian_log.json"))
        logger._path = self.json_log

        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.subscribe(Events.OPTIMIZATION_END, logger)

        _logger = ScreenLogger(2)
        optimizer.subscribe(Events.OPTIMIZATION_START, _logger)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, _logger)
        optimizer.subscribe(Events.OPTIMIZATION_END, _logger)

        # "When dealing with functions with discrete parameters,or particularly erratic target space it
        #  might be beneficial to increase the value of the alpha parameter.
        #  This parameters controls how much noise the GP can handle,
        #  so increase it whenever you think that extra flexibility is needed."
        optimizer.maximize(init_points=20, n_iter=160, alpha=1e-2, n_restarts_optimizer=0)

        print(optimizer.max)
        print("\nScore", optimizer.max["target"] * -1)
        for key, val in self.normalize_params(optimizer.max['params']).items():
            print("\t", key, val)

    def load_norm_weights(self) -> Dict[str, float]:
        norm_weights = dict()
        with open(self.norm_weights_file, 'r') as norm_weights_in:
            norm_weights_in.readline()[:-1].split(",")  # Skip header
            for line in norm_weights_in:
                group = re.findall(r'([\w\[\],;>%]+),([\d.]+)', line)
                if group:
                    norm_weights[group[0][0]] = float(group[0][1])
        return norm_weights

    def store_fitness_guess(self, x):
        x = self.normalize_params(x)
        self.run_configuration['x'] = x
        self.run_configuration['norm_schedule'] = NormSchedule(x, "2020-06-28")
        self.run_configuration['policy_schedule_name'] = os.path.join(
            get_project_root(),
            '.persistent',
            'policies',
            # TODO: FIx
            "norm-schedule-policy-EO0_{EO0_start}_{EO0_duration}-"
            "EO1_{EO1_start}_{EO1_duration}-EO2_{EO2_start}_{EO2_duration}-"
            "EO3_{EO3_start}_{EO3_duration}-EO4_{EO4_start}_{EO4_duration}-"
            "EO5_{EO5_start}_{EO5_duration}-EO6_{EO6_start}_{EO6_duration}-"
            "EO7_{EO7_start}_{EO7_duration}-EO8_{EO8_start}_{EO8_duration}.csv".format(**x)
        )

        if not os.path.exists(self.csv_log):
            with open(self.csv_log, "a") as fout:
                fout.write(
                    "fips,#counties,score,x,time_finished,calibration_start_time\n"
                )

    def prepare_simulation_run(self, x):
        self.run_configuration['norm_schedule'].write_to_file(self.run_configuration['policy_schedule_name'])

    def score_simulation_run(self, x: Dict[str, float], directories: List[Dict[int, str]]) -> float:
        fitness = 0
        for norm in self.norm_counts.keys():
            active_duration = self.run_configuration['norm_schedule'].get_active_duration(norm)
            affected_agents = self.find_number_of_agents_affected_by_norm(norm, directories)
            norm_weight = self.norm_weights[norm]
            fitness += (active_duration * norm_weight * affected_agents)
        return self.count_infected_agents(directories) + (self.societal_global_impact_weight * fitness)

    def _write_csv_log(self, score):
        pass

    def find_number_of_agents_affected_by_norm(self, norm_name: str, directories: List[Dict[int, str]]) -> int:
        if "StayHomeSick" in norm_name:
            return self.find_number_of_agents_affected_by_stayhome_if_sick(directories)
        else:
            return self.norm_counts[norm_name]['affected_agents']

    def find_number_of_agents_affected_by_stayhome_if_sick(self, directories: List[Dict[int, str]]) -> int:
        # TODO, just report number of symptomatically ill people
        # TODO How to find people who were symptomatic?
        return round(.6 * self.count_infected_agents(directories))

    def count_infected_agents(self, directories: List[Dict[int, str]]) -> int:
        amounts = defaultdict(int)
        for node in directories:
            for run, directory in node.items():
                with open(os.path.join(directory, 'epicurve.pansim.csv'), 'r') as file_in:
                    headers = file_in.readline()[:-1].split(",")
                    values = file_in.readlines()[-1][:-1].split(",")
                    infected = sum(map(lambda x: int(values[headers.index(x)]), ["isymp", "iasymp", "recov"]))
                    amounts[run] += infected
        return round(np.average(list(amounts.values())))

    @staticmethod
    def print_data_point_ranges() -> None:
        """
        Quickly print the dates corresponding to the allowed range of data points
        """
        date = datetime.datetime(year=2020, month=3, day=1)
        data_point = 0
        while date < datetime.datetime(year=2020, month=7, day=1):
            data_point += 1
            date = datetime.datetime.strptime(date_from_data_point(data_point), "%Y-%m-%d")

    def load_norm_application_counts(self):
        filename = os.path.abspath(
            os.path.join(
                get_project_root(),
                '.persistent',
                "affected-agents-per-norm-{0}.csv".format("-".join(sorted(map(lambda x: str(self.counties[x]['fipscode']), self.counties))))
            )
        )

        if not os.path.exists(filename):
            print("Using behavior model to find what number of agents is affected by what norms")
            self.get_extra_java_commands = lambda: ['--count-affected-agents']
            base_dir_func_backup = self.get_base_directory
            self.get_base_directory = lambda: os.path.abspath(os.path.join(get_project_root(), '.persistent'))
            java_command = self._java_command()
            java_command = java_command[:java_command.index('2>&1')]
            subprocess.run(java_command, stderr=subprocess.PIPE)
            self.get_extra_java_commands = lambda: []
            self.get_base_directory = base_dir_func_backup

        norm_counts = dict()
        with open(filename, 'r') as file_in:
            file_in.readline()  # Skip header
            for line in file_in:
                match = re.findall(r'(\w+(?:\[[\w,;>%]+])?);(\d+);(\d+)', line)
                if match:
                    norm_counts[match[0][0]] = {'affected_agents': int(match[0][1]), 'affected_duration': int(match[0][2])}

        return norm_counts


def test_data_point_range():
    EOOptimization.print_data_point_ranges()
    print("Done")


def test_fitness():
    opt = EOOptimization()


if __name__ == "__main__":
    # test_fitness()
    for x in range(9):
        print(f"EO{x}_{{x[EO{x}_start]}}_{{x[EO{x}_duration]}}-", end="")
