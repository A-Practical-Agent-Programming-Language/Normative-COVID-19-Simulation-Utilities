import os
import re
import sys
from collections import defaultdict
from math import sqrt
from typing import Dict, Any, List, Tuple

from sklearn.metrics import mean_squared_error

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
            get_project_root(), "va-counties-estimated-covid19-cases.csv"
        ),
    ):
        self.counties = counties
        self.__epicurve_filename = epicurve_filename
        self.case_file = case_file
        self.county_case_data = self.__load_case_data()
        self.baseline = self.__create_relevant_epicurve()

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
                    target.append(self.baseline[date])

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

        return sqrt(mean_squared_error(target, predicted))

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


if __name__ == "__main__":
    t = load_toml_configuration(sys.argv[1])

    os.chdir("../")
    e = EpicurveRMSE(t["counties"])

    print(e.calculate_rmse([dict(zip(range(len(sys.argv[2:])), sys.argv[2:]))]))
