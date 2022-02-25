import datetime
import os
from typing import List, Dict

import toml

from classes.ExecutiveOrderOptimizer import NormService
from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule
from classes.execution.CodeExecution import CodeExecution


class NormExperiment(CodeExecution):
    rundirectory_template = [
        "experiment",
        "{ncounties}counties-fips-{fips}",
        "experiment-{experiment_index}-norms-until{experiment_max_date}-run{run}",
    ]
    norm_schedule_base = os.path.join(".persistent", ".tmp", "norms_until_dates")
    norm_schedule_dir = os.path.join(norm_schedule_base, "norm-schedules")
    county_config_dir = os.path.join(norm_schedule_base, "county-configuration")

    def __init__(self, *args, **kwargs):
        super(NormExperiment, self).__init__(*args, **kwargs)
        os.makedirs(self.norm_schedule_dir, exist_ok=True)
        os.makedirs(self.county_config_dir, exist_ok=True)
        self.target_file = "tick-averages.csv"
        self.original_county_configuration_file = self.county_configuration_file
        self.end_date = self.get_simulation_end_date()
        self.norms_file = self.get_norm_schedule()
        self.experiment_dates = self.get_experiment_dates(self.norms_file, self.end_date)
        print(f"Found {len(self.experiment_dates)} unique dates")

    def initiate(self):
        for i, date in enumerate(self.experiment_dates):
            norm_schedule = self.create_norm_schedule_for_date(date)
            self.county_configuration_file = self.update_county_configuration_file(
                date, norm_schedule
            )
            self.run_configuration["experiment_index"] = i
            self.run_configuration["experiment_max_date"] = date
            print(
                f"Starting {self.n_runs} runs for all norms up to and including {date}), using {self.county_configuration_file}"
            )
            self.calibrate(None)

    def run_by_removing_norms(self):
        self.update_schedule_directories("experiment-without-{removed_norm}-run{run}", "norm-schedules-with-specific-norms-removed")

        expected = [1 for EO, norm_list in NormService.norms.items() for _ in norm_list]
        print(f"Expecting {sum(expected) * self.n_runs} simulations")

        for EO, norm_list in NormService.norms.items():
            for norm in norm_list:
                norm_str = norm[1]
                if norm[2]:
                    norm_str += '-' + norm[2].replace(",", "-for-").replace(";", "-and-")
                norm_str = norm_str.replace(" ", "").replace("%", "pct").replace(">", "-over-")
                print(norm_str)
                self.run_configuration["removed_norm"] = norm_str
                ns = NormSchedule.from_norm_schedule(self.norms_file, self.last_expected_simulation_date)
                ns.remove_norms(norm)

                norm_schedule_file = os.path.join(self.norm_schedule_dir, f"norm-schedule-without-{norm_str}.csv")
                ns.write_to_file(norm_schedule_file)
                self.county_configuration_file = self.update_county_configuration_file(
                    "without-" + norm_str,
                    norm_schedule_file
                )
                self.calibrate(None)

    def run_by_removing_EOs(self):
        self.update_schedule_directories("experiment-without-{removed_eo}-run{run}", "norm-schedules-with-specific-EOs-removed")
        for EO, norm_list in NormService.norms.items():
            self.run_configuration["removed_eo"] = EO
            ns = NormSchedule.from_norm_schedule(self.norms_file, self.last_expected_simulation_date)
            ns.remove_norms(norm_list)

            norm_schedule_file = os.path.join(self.norm_schedule_dir, f"norm-schedule-without-{EO}.csv")
            ns.write_to_file(norm_schedule_file)
            self.county_configuration_file = self.update_county_configuration_file(
                "without-" + EO, norm_schedule_file
            )
            self.calibrate(None)

    def update_schedule_directories(self, run_directory_format, norm_schedule_format):
        self.rundirectory_template = [
            self.output_dir,
            "experiment-remove-EOs",
            "{ncounties}counties-fips-{fips}",
            run_directory_format
        ]
        self.norm_schedule_base = os.path.join(".persistent", ".tmp", norm_schedule_format)
        self.norm_schedule_dir = os.path.join(self.norm_schedule_base, "norm-schedules")
        self.county_config_dir = os.path.join(self.norm_schedule_base, "county-configuration")
        os.makedirs(self.norm_schedule_dir, exist_ok=True)
        os.makedirs(self.county_config_dir, exist_ok=True)

    def store_fitness_guess(self, x):
        pass

    def prepare_simulation_run(self, x):
        pass

    def score_simulation_run(self, x, directories: List[Dict[int, str]]) -> float:
        pass

    def _write_csv_log(self, score):
        pass

    @staticmethod
    def get_experiment_dates(norms_file, experiment_end_date):
        experiment_dates = ["0000-00-00"]  # Start with empty norms
        with open(norms_file, "r") as norms_in:
            for line in norms_in:
                if len(line.split(",")):
                    date = line.split(",")[0]
                    if date < str(experiment_end_date):
                        experiment_dates.append(date)

        return sorted(list(set(experiment_dates)))

    def get_norm_schedule(self):
        return toml.load(self.original_county_configuration_file)["simulation"]["norms"]

    def get_simulation_end_date(self):
        conf = toml.load(self.original_county_configuration_file)
        if "iterations" in conf["simulation"]:
            if isinstance(conf["simulation"]["iterations"], datetime.date):
                return conf["simulation"]["iterations"] + datetime.timedelta(days=1)
            else:
                if "startdate" in conf["simulation"]:
                    return conf["simulation"]["startdate"] + datetime.timedelta(
                        days=int(conf["simulation"]["iterations"])
                    )

        return datetime.date(9999, 1, 1)

    def update_county_configuration_file(self, date, norm_schedule_file):
        conf = toml.load(self.original_county_configuration_file)
        conf["simulation"]["norms"] = os.path.abspath(norm_schedule_file)
        new_file_location = os.path.join(
            self.county_config_dir, f"county_config_until_{date}.csv"
        )
        with open(new_file_location, "w") as conf_out:
            toml.dump(conf, conf_out)
        return new_file_location

    def create_norm_schedule_for_date(self, date):
        output_file = os.path.join(
            self.norm_schedule_dir, f"norm_schedule_until_{date}.csv"
        )
        has_norms = False
        with open(self.norms_file, "r") as norms_in:
            with open(output_file, "w") as norms_out:
                norms_out.write(norms_in.readline())
                for line in norms_in:
                    if not len(line.split(",")) or line.split(",")[0] <= date:
                        has_norms = True
                        norms_out.write(line)

        return output_file if has_norms else ""

    def _process_loss(self, x, loss):
        """
        The experiment execution does not use the calculated loss, so we can return None, but the _process_loss method
        of super uses the numbers (None in this class) returned by score_simulation_run().
        Let's just pass that execution entirely
        """
        pass
