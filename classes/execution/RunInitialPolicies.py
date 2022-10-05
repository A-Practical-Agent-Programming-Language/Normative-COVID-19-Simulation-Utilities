import os
import re
import shutil
from typing import List, Dict

import toml

from classes.execution.CodeExecution import CodeExecution


class RunInitialPolicies(CodeExecution):

    rundirectory_template = [
        "optimization",
        "initial-policies",
        "policy-{policy-index}-run-{run}"
    ]

    progress_format = (
        "[INITIAL_POLICIES]: {time}] {ncounties} counties ({fips}): policy {policy-index} (dir={output_dir})\n"
    )

    def __init__(self, initial_policies_location, *args, **kwargs):
        super(RunInitialPolicies, self).__init__(*args, **kwargs)
        self.original_county_configuration_file = self.county_configuration_file
        self.target_file = "tick-averages.csv"
        self.initial_policies_location = initial_policies_location
        self.schedules = self.find_norm_schedules()

    def run_simulations(self):
        for schedule in self.schedules:
            self.run_configuration["policy-index"] = schedule["index"]
            self.run_configuration["norm-schedule-file"] = schedule["schedule"]
            self.run_configuration["norm-schedule-name"] = os.path.basename(schedule["name"])
            self.run_configuration["policy_configuration"] = schedule["protocol"]

            self.calibrate(None)

    def find_norm_schedules(self):
        schedules = list()
        if "norm-schedule" in os.listdir(self.initial_policies_location):
            base_path = os.path.join(self.initial_policies_location, "norm-schedule")
        else:
            base_path = self.initial_policies_location

        for ns in os.listdir(base_path):
            potential_norm_schedule = os.path.join(base_path, ns)
            if self.test_is_norm_schedule(potential_norm_schedule):
                s = dict(
                    schedule=potential_norm_schedule,
                    name=ns.replace(".csv", ""),
                    index=int(re.sub(r'\D', '', ns))
                )

                potential_protocol_file = os.path.join(self.initial_policies_location, "protocol", ns.replace(".csv", ".json"))
                if os.path.exists(os.path.join(potential_protocol_file)):
                    s["protocol"] = potential_protocol_file
                schedules.append(s)

        all_indices = [s['index'] for s in schedules]
        assert len(set(all_indices)) == len(all_indices), "Norm indices contain duplicates"

        return schedules

    def store_fitness_guess(self, x):
        pass

    def prepare_simulation_run(self, x):
        os.makedirs(self.get_base_directory(), exist_ok=True)
        self.county_configuration_file = \
            self.update_county_configuration_file(self.run_configuration["norm-schedule-file"])
        for f in [
            self.run_configuration["norm-schedule-file"],
            self.run_configuration["policy_configuration"]
        ]:
            if os.path.exists(f):
                shutil.copyfile(f, os.path.join(self.get_base_directory(), os.path.basename(f)))

    def _process_loss(self, x: tuple, loss: float):
        pass

    def score_simulation_run(self, x: tuple, directories: List[Dict[int, str]]) -> float:
        pass

    def _write_csv_log(self, score):
        pass

    @staticmethod
    def test_is_norm_schedule(path):
        if not path.endswith(".csv"):
            return False
        with open(path) as potential_norm_schedule:
            return potential_norm_schedule.readline().startswith("start,norm,end,param")

    def update_county_configuration_file(self, norm_schedule):
        conf = toml.load(self.original_county_configuration_file)
        conf["simulation"]["norms"] = os.path.abspath(norm_schedule)
        new_file_location = os.path.join(self.get_base_directory(), "county-configuration.toml")
        with open(new_file_location, "w") as conf_out:
            toml.dump(conf, conf_out)
        return new_file_location


