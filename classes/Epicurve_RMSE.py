import os
import re
import sys
from collections import defaultdict
from math import sqrt
from typing import Dict, Any, List, Tuple

import click
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils import check_consistent_length

from utility.utility import load_toml_configuration, get_project_root

Date = str
Fips = int
Run = int


class EpicurveRMSE(object):
    """
    Class to calculate the RMSE between epicurve produced by the simulation with actual case data
    """

    def __init__(
        self,
        counties: [Dict[str, Dict[str, Any]]],
        epicurve_filename: str = "epicurve.sim2apl.csv",
        case_file: str = os.path.join(
            get_project_root(), "external", "va-counties-estimated-covid19-cases.csv"
        ),
        scale: None or float=None,
        loss_function=mean_squared_error
    ):
        self.counties = counties
        self.__epicurve_filename = epicurve_filename
        self.case_file = case_file
        self.county_case_data = self.__load_case_data()
        self.baseline = self.__create_relevant_epicurve()
        self.scale = scale
        self.loss_function = loss_function

    def __load_case_data(self) -> Dict[int, Dict[Date, int]]:
        """
        Load the case data available for the state
        Returns:
                Dictionary of dictionaries, where the outer key is the FIPS code of the county for which case data is
                known, the inner key is the date, and the value is the number of recorded cases until that day
        """
        epicurve = dict()

        with open(self.case_file, "r") as cases_in:
            headers = cases_in.readline()[:-1].split(",")
            for line in cases_in:
                data = line[:-1].split(",")
                date = data[headers.index("date")]
                fips_raw = data[headers.index("fips")]
                if fips_raw.lower() in ["", "Unknown"]:
                    continue
                fips = int(re.match(r"(?:510?)?(\d{2,3})", fips_raw).groups()[0])
                cases = data[headers.index("cases")]

                if fips not in epicurve:
                    epicurve[fips] = dict()
                epicurve[fips][date] = float(cases)

        return epicurve

    def __create_relevant_epicurve(self):
        """
        Create one epicurve from the per-county case data, by summing the cases on each day for all counties
        participating in the simulation

        Returns:
                Dictionary with dates as key, and cumulative case counts for all counties in the simulation as value
        """
        epicurve = defaultdict(int)
        for county in self.counties.values():
            if county["fipscode"] in self.county_case_data:
                for d in self.county_case_data[county["fipscode"]]:
                    epicurve[d] += self.county_case_data[county["fipscode"]][d]

        if not len(epicurve):
            raise ValueError("No epicurve found for provided counties!")

        return epicurve

    def calculate_rmse(self, run_directories: List[Dict[Run, str]]) -> float:
        """
        Calculates the root mean squared error (RMSE) between the number of recovered agents in the simulation and
        the number of actually observed cases to account for testing
        uncertainty)

        Args:
                run_directories:    Output directories of simulation runs for this parameter configuration

        Returns:
                Double: RMSE between actual case count and number of agents recovered in the simulation

        """
        predicted_recovered_list: List[Tuple[Run, Dict[Date, int]]] = [
            (
                run,
                self.__read_all_infected_from_epicurve(
                    os.path.join(run_directory[run], self.__epicurve_filename)
                ),
            )
            for run_directory in run_directories
            for run in run_directory
        ]

        # Add recovered numbers from files spread over multiple compute nodes, but keep runs separate
        dates = set(self.baseline.keys())
        predicted_combined: Dict[Run, Dict[Date, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for run, values in predicted_recovered_list:
            for date, amount in values.items():
                dates.add(date)
                predicted_combined[run][date] += amount
        dates = list(sorted(dates))

        # Create the predicted/target lists for RMSE
        predicted, target = list(), list()
        for run, predicted_for_run in predicted_combined.items():
            for date in dates:
                if date in self.baseline and date in predicted_for_run:
                    predicted.append(predicted_for_run[date])
                    target.append(self.baseline[date] if self.scale is None else self.baseline[date] * self.scale)

        # Write values used for calculating RSME to file, so plot of fits can be created later
        for run_directory in run_directories:
            for run in run_directory:
                with open(
                    os.path.join(run_directory[run], "compare-case-data.csv"), "w"
                ) as epicurve_out:
                    epicurve_out.write("Date,Cases,Recovered\n")
                    for date in dates:
                        cases = self.baseline[date] if date in self.baseline else ""
                        recovered = (
                            predicted_combined[run][date]
                            if date in predicted_combined[run]
                            else ""
                        )
                        epicurve_out.write(
                            f"{date},{cases},{recovered}\n"
                        )

        return sqrt(self.loss_function(target, predicted))

    @staticmethod
    def __read_all_infected_from_epicurve(run_directory: str) -> Dict[Date, int]:
        epicurve = dict()
        with open(run_directory) as epicurve_in:
            headers = epicurve_in.readline()[:-1].split(";")
            required_headers = ["EXPOSED", "INFECTED_SYMPTOMATIC", "INFECTED_ASYMPTOMATIC", "RECOVERED"]
            required_indices = [headers.index(h) for h in required_headers]
            for line in epicurve_in:
                data = line[:-1].split(";")
                all_infected = sum([int(data[x]) for x in required_indices])
                epicurve[data[headers.index("Date")]] = all_infected

        return epicurve


def percentage_mean_square_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", squared=True):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    check_consistent_length(y_true, y_pred, sample_weight)
    output_errors = np.average((y_true - y_pred / y_true) ** 2, axis=0, weights=sample_weight)

    if not squared:
        output_errors = np.sqrt(output_errors)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


@click.command()
@click.option(
    "--simulation-output",
    "-s",
    help="Directory containing output of one simulation, or containing multiple simulation configurations (if so, all simulations in directory will be plotted",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
    required=True,
)
@click.option(
    "--county-configuration",
    "-c",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
    help="Specify the county configuration used for this simulation run containing all simulated counties",
    required=True,
)
@click.option("--percentage-instead/--no-percentage-instead", help="Calculate RMSE from percentage difference", default=False)
@click.option(
    "--scale-factor",
    default=None,
    type=float,
    required=False
)
@click.option(
    "--projections/--no-projections",
    help="Use projected case values as target instead of confirmed cases?",
    default=False
)
def calculate_RMSE_on_run(county_configuration, simulation_output, projections, scale_factor, percentage_instead):
    t = load_toml_configuration(county_configuration)
    os.chdir("../")
    real_cases = os.path.join(
        get_project_root(),
        "external",
        f"va-counties-{'estimated-' if projections else ''}covid19-cases.csv",
    )
    e = EpicurveRMSE(
        t["counties"],
        case_file=real_cases,
        scale=scale_factor,
        loss_function=percentage_mean_square_error if percentage_instead else mean_squared_error
    )
    if "epicurve.sim2apl.csv" in os.listdir(simulation_output):
        runs = {0: simulation_output}
    else:
        runs = dict()
        index = 0
        for subdir in os.listdir(simulation_output):
            abs_sub_dir = os.path.join(simulation_output, subdir)
            if os.path.isdir(abs_sub_dir) and "epicurve.sim2apl.csv" in os.listdir(abs_sub_dir):
                runs[index] = abs_sub_dir
                index += 1

    print(e.calculate_rmse([runs]))


if __name__ == "__main__":
    calculate_RMSE_on_run()
