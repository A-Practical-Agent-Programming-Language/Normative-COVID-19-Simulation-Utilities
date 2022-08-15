import os
import shutil
from typing import List, Dict

import toml

from classes.execution.CodeExecution import CodeExecution
from utility.utility import get_project_root


class RunWithNormSchedules(CodeExecution):
    rundirectory_template = [
        "run-with-norm-schedules",
        "{ncounties}counties-fips-{fips}",
        "{norm-schedule-name}",
        "run{run}"
    ]
    progress_format = (
        "[REPEAT: {time}] {ncounties} counties ({fips}): {score} (dir={output_dir})\n"
    )

    def __init__(self, norm_schedules, *args, **kwargs):
        super(RunWithNormSchedules, self).__init__(*args, **kwargs)
        self.original_county_configuration_file = self.county_configuration_file
        self.target_file = "tick-averages.csv"
        self.norm_schedules = norm_schedules
        self.run_simulations()

    def run_simulations(self):
        for norm_schedule in self.norm_schedules:
            self.run_configuration["norm-schedule-file"] = norm_schedule
            self.run_configuration["norm-schedule-name"] = os.path.basename(norm_schedule).replace(".csv", "")
            self.county_configuration_file = self.update_county_configuration_file(norm_schedule)

            self.calibrate(None)

    def store_fitness_guess(self, x):
        pass

    def prepare_simulation_run(self, x):
        os.makedirs(self.get_base_directory(), exist_ok=True)
        norm_schedule = self.run_configuration["norm-schedule-file"]
        shutil.copyfile(norm_schedule, os.path.join(self.get_base_directory(), os.path.basename(norm_schedule)))

    def score_simulation_run(self, x: tuple, directories: List[Dict[int, str]]) -> float:
        pass

    def _write_csv_log(self, score):
        pass

    def update_county_configuration_file(self, norm_schedule):
        conf = toml.load(self.original_county_configuration_file)
        conf["simulation"]["norms"] = os.path.abspath(norm_schedule)
        new_file_location = get_project_root(
            ".persistent", ".tmp", "conf_with_" + self.run_configuration["norm-schedule-name"] + ".toml"
        )
        with open(new_file_location, "w") as conf_out:
            toml.dump(conf, conf_out)
        return new_file_location


