import datetime
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List

import click
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from classes.Epicurve_RMSE import EpicurveRMSE
from utility import run_finder
from utility.plots import metrics, add_norms_to_graph
from utility.run_finder import find_disease_runs
from utility.utility import load_toml_configuration, get_project_root

Path = os.PathLike or str
Date = str  # Date formatted as YYYY-MM-DD


@click.command()
@click.option(
    "--simulation-output",
    "-s",
    help="Directory containing output of one simulation, or containing multiple simulation configurations (if so, all simulations in directory will be plotted",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
    required=True,
)
@click.option(
    "--real-cases",
    "-c",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
    help="Specify the file with the ground truth of real cases in this state",
    default=os.path.join(
        get_project_root(), "external", "va-counties-estimated-covid19-cases.csv"
    ),
    required=False,
)
@click.option(
    "--county-configuration",
    "-c",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
    help="Specify the county configuration used for this simulation run containing all simulated counties",
    required=True,
)
@click.option(
    "--table-output",
    "-p",
    type=str,
    default=None,
    help="If explicitly set to true, the script will create a file containing a table with the mobility data with "
         "the file name as passed to this argument, and will print Tikz picture code to plot that data in LaTeX",
)
@click.option(
    "--is-experiment",
    "-e",
    type=bool,
    default=False,
    help="If this is a directory containing multiple experiment norms, pass true to this argument. The run configuration"
    "will not be parsed from the simulation output directory names in that instance.",
)
@click.option(
    "--scale-factor",
    type=int,
    default=30,
    help="If the scale factor can not be extracted from the simulation output directory name, what scale"
    "factor should be used? (this value will be ignored if at least one of the simulation output directory"
    "names contains a scale factor value)",
)
def start_plot(**args):
    EpicurvePlotter(**args)


class EpicurvePlotter(object):
    """
    Can plot the number of recovered agents for one or more simulation runs (with the same parameter configuration)
    against the number of actually *recorded* cases (scaled by the scale factor used in the calibration process)
    into a single plot, together with the activation and deactivation of norms.

    Can also produce a LaTeX table for use in publications

    See:
        plot_epicurve_average.py --help
    """

    def __init__(
        self,
        simulation_output,
        county_configuration,
        real_cases,
        table_output,
        scale_factor,
        is_experiment,
    ):
        self.simulation_output = simulation_output
        self.counties = load_toml_configuration(county_configuration)
        self.epicurve_RMSE = EpicurveRMSE(
            self.counties["counties"], case_file=real_cases
        )

        if is_experiment:
            merged_curves, max_agents = self.plot_unknown_experiment_runs(scale_factor)
        else:
            try:
                merged_curves, max_agents, scale_factor = self.plot_extracted_calibration_runs()
            except Exception as e:
                print(
                    "Failed to create plots. Is this the output directory of an experiment, which does not contain"
                    "information about the run configuration employed? Then make sure to set --is-experiment to true,"
                    "and also pass the scale factor using --scale-factor"
                )
                print("The expection was:")
                raise e

        if table_output is not None:
            self.create_data_table(scale_factor, merged_curves, max_agents, table_output)

    def plot_extracted_calibration_runs(
        self,
    ) -> (Dict[any, Dict[Date, Dict[str, List[int]]]], int):
        merged_curves = dict()
        max_agents = 0
        configurations = find_disease_runs(self.simulation_output)
        for configuration, run_group in configurations.items():
            rmse_group = run_finder.runs_to_rmse_list(run_group)
            rmse = self.epicurve_RMSE.calculate_rmse(configuration[2], rmse_group)
            print("RMSE", rmse, configuration)
            isymp, iasymp, scale_factor = (
                configuration[0],
                configuration[1],
                configuration[2],
            )
            curves, agents = zip(*[self.read_epicurve_file(x) for x in run_group])
            merged_curve = self.merge_curves(curves)
            self.plot_merged_curves(
                run_group[0],
                scale_factor,
                merged_curve,
                max(agents),
                "Calibrated disease model",
                f"Score: {rmse:.2f} -- {isymp:.{2}e} symptomatic, {iasymp:.{2}e} asymptomatic",
                os.path.join(
                    self.simulation_output,
                    f"epicurve-{rmse}rmse-isymp{isymp:.3e}-iasymp-{iasymp:.3e}.png",
                ),
            )
            merged_curves[configuration] = merged_curve
            max_agents = max(max_agents, max(agents))

        return merged_curves, max_agents, scale_factor

    def plot_unknown_experiment_runs(
        self, scale_factor: int
    ) -> (Dict[any, Dict[Date, Dict[str, List[int]]]], int):
        merged_curves = dict()
        max_agents = 0

        grouped_runs = self.group_runs()
        for name, run_group in grouped_runs.items():
            curves, agents = zip(
                *[self.read_epicurve_file(run_group[x]) for x in run_group]
            )
            merged_curve = self.merge_curves(curves)
            merged_curves[name] = merged_curve
            self.plot_merged_curves(
                run_group[list(run_group.keys())[0]],
                scale_factor,
                merged_curve,
                max(agents),
                name,
                name,
                os.path.join(self.simulation_output, f"{name}.png"),
            )
            max_agents = max(max_agents, max(agents))

        return merged_curves, max_agents

    def group_runs(self) -> Dict[str, Dict[int, str]]:
        if os.path.exists(os.path.join(self.simulation_output, "epicurve.sim2apl.csv")):
            # Directory is straight simulation output, nothing to group
            return {'': {0: self.simulation_output}}
        grouped_runs = defaultdict(dict)
        for path in os.listdir(self.simulation_output):
            if (
                os.path.isdir(os.path.join(self.simulation_output, path))
                and "run" in path
            ):
                match = re.findall(r"(.*)-run(\d+)$", path)
                if len(match):
                    name, run = match[0]
                    grouped_runs[name][run] = os.path.join(self.simulation_output, path)
        return grouped_runs

    @staticmethod
    def read_epicurve_file(
        simulation_output_directory: Path,
    ) -> (Dict[Date, Dict[str, int or str]], int):
        """
        Reads the epicurve file from a specific simulation output directory
        Args:
            simulation_output_directory: Full path to epicurve.sim2apl.csv file to read

        Returns:
            First element: Dictionary with dates as keys, and dictionary of key-value pairs with simulated cases by day
            Second element: The maximum number of agents recorded on any specific day
        """
        curve = dict()
        max_agents = 0
        with open(
            os.path.join(simulation_output_directory, "epicurve.sim2apl.csv"), "r"
        ) as file_in:
            headers = file_in.readline()[:-1].split(";")
            for line in file_in:
                line = dict(
                    zip(
                        headers,
                        [int(x) if "-" not in x else x for x in line.split(";")],
                    )
                )
                numbers = [
                    "NOT_SET",
                    "SUSCEPTIBLE",
                    "EXPOSED",
                    "INFECTED_SYMPTOMATIC",
                    "INFECTED_ASYMPTOMATIC",
                    "RECOVERED",
                ]
                m = sum([line[x] for x in numbers])
                max_agents = max(max_agents, m)
                curve[line["Date"]] = line

        return curve, max_agents

    @staticmethod
    def merge_curves(
        curves: List[Dict[Date, Dict[str, int or str]]]
    ) -> Dict[Date, Dict[str, List[int]]]:
        curve_values = dict()
        for curve in curves:
            for day in curve:
                if day not in curve_values:
                    curve_values[day] = dict(
                        susceptible=[], infected=[], recovered=[], calibration_target=[], total=[]
                    )
                susc = int(curve[day]["SUSCEPTIBLE"])
                inf = (
                    int(curve[day]["EXPOSED"])
                    + int(curve[day]["INFECTED_SYMPTOMATIC"])
                    + int(curve[day]["INFECTED_ASYMPTOMATIC"])
                )
                rec = int(curve[day]["RECOVERED"])
                calibration_target = inf + rec
                curve_values[day]["susceptible"].append(susc)
                curve_values[day]["infected"].append(inf)
                curve_values[day]["recovered"].append(rec)
                curve_values[day]["total"].append(susc + inf + rec)
                curve_values[day]["calibration_target"].append(calibration_target)

        return curve_values

    def get_case_data(self) -> Dict[Date, int]:
        fips_codes = [
            self.counties["counties"][county]["fipscode"]
            for county in self.counties["counties"]
        ]
        combined_observed_cases = defaultdict(int)
        for fips in fips_codes:
            for date, cases in self.epicurve_RMSE.county_case_data[fips].items():
                combined_observed_cases[date] += cases
        return combined_observed_cases

    def plot_merged_curves(
        self,
        sample_simulation_dir: str,
        scale_factor: int,
        curves: Dict[Date, Dict[str, List[int]]],
        max_agents: int,
        title: str,
        subtitle: str,
        plot_name: str = None,
    ) -> None:

        fig, ax = plt.subplots()

        days = sorted(curves.keys())
        cases = self.get_case_data()

        recovered = [curves[day]["calibration_target"] for day in days]
        recorded = [
            cases[day] * scale_factor if day in cases else math.nan for day in days
        ]

        recovered_pct = [[r / max_agents * 100 for r in lst] for lst in recovered]
        recorded_pct = [r / max_agents * 100 for r in recorded]

        x = np.arange(0, len(days), 1)

        y_recc, y_recc_err = metrics(recovered_pct)
        ax.plot(x, y_recc, "k", color="#ffd320", label="Simulated")
        ax.fill_between(
            x,
            y_recc - y_recc_err,
            y_recc + y_recc_err,
            alpha=0.2,
            facecolor="#ffd320",
            antialiased=True,
        )

        ax.plot(x, recorded_pct, color="blue", label=f"Real cases * {scale_factor}")
        add_norms_to_graph(ax, days, simulation_output_dir=sample_simulation_dir)

        plt.suptitle(title)
        plt.title(subtitle)
        plt.legend()
        plt.grid(which="both", axis="both")
        plt.xlim([0, len(days)])
        plt.ylim([ax.dataLim.ymin, ax.dataLim.ymax])

        plt.xticks(x, map(lambda y: y[5:], days), rotation=90)
        ax.xaxis.set_major_locator(MultipleLocator(7))
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylabel("Number of agents")
        ax.set_xlabel("Simulation day (month-day in 2020)")
        plt.tight_layout()
        if plot_name is not None:
            plt.savefig(plot_name, dpi=300)
        plt.show()

    def create_data_table(
        self,
        scale_factor: int,
        merged_curves: Dict[any, Dict[Date, Dict[str, List[int]]]],
        max_agents: int,
        table_output: str
    ):
        start_date = datetime.date(2020, 1, 1)
        end_date = datetime.date(2020, 12, 1)
        delta = datetime.timedelta(days=1)

        recovered_cases = dict()
        cases = self.get_case_data()

        for name, curves in merged_curves.items():
            days = sorted(curves.keys())
            rec_raw = [curves[day]["recovered"] for day in days]
            recovered, recovered_err = metrics(rec_raw)
            rec_pct = [[x / max_agents * 100 for x in day] for day in rec_raw]
            recovered_pct, recovered_err_pct = metrics(rec_pct)
            recovered_cases[name] = dict(
                recovered=dict(zip(days, recovered)),
                y_err=dict(zip(days, recovered_err)),
                recovered_pct=dict(zip(days, recovered_pct)),
                recovered_pct_err=dict(zip(days, recovered_err_pct)),
                cases=dict(
                    zip(
                        days,
                        [
                            (cases[day] * scale_factor) if day in cases else 0
                            for day in days
                        ],
                    )
                ),
                cases_pct=dict(
                    zip(
                        days,
                        [
                            (cases[day] * scale_factor) / max_agents * 100
                            if day in cases
                            else 0
                            for day in days
                        ],
                    )
                ),
            )

        headers = [
            "x",
            "date",
            f"cases_times_{scale_factor}",
            f"cases_times_{scale_factor}_pct",
        ]
        experiments = sorted(list(merged_curves.keys()))
        headers += [
            x
            for sublist in [
                [
                    name,
                    f"{name}__inf",
                    f"{name}__sup",
                    f"{name}_pct",
                    f"{name}_pct__inf",
                    f"{name}_pct__sup",
                ]
                for name in map(lambda x: "-".join(map(str, x)), experiments)
            ]
            for x in sublist
        ]

        index = 0
        y_values = []
        y_values_pct = []
        cases = []
        cases_pct = []

        table_name = table_output if table_output.endswith(".tex") else f"{table_output}.tex"
        with open(os.path.join(self.simulation_output, table_name), "w") as results_out:
            results_out.write("\t".join(headers) + "\n")
            while start_date < end_date and index <= 182:
                if str(start_date) in recovered_cases[name]["cases"]:
                    day_cases = recovered_cases[name]["cases"][str(start_date)]
                    day_cases_pct = recovered_cases[name]["cases_pct"][str(start_date)]
                    cases.append(day_cases)
                    cases_pct.append(day_cases_pct)
                else:
                    day_cases, day_cases_pct = 0, 0
                values = [index, str(start_date), day_cases, day_cases_pct]
                for name in experiments:
                    if str(start_date) in recovered_cases[name]["recovered"]:
                        r = recovered_cases[name]["recovered"][str(start_date)]
                        inf = r + recovered_cases[name]["y_err"][str(start_date)]
                        sup = r - recovered_cases[name]["y_err"][str(start_date)]
                        r_pct = recovered_cases[name]["recovered_pct"][str(start_date)]
                        inf_pct = (
                            r_pct
                            + recovered_cases[name]["recovered_pct_err"][
                                str(start_date)
                            ]
                        )
                        sup_pct = (
                            r_pct
                            - recovered_cases[name]["recovered_pct_err"][
                                str(start_date)
                            ]
                        )
                        values += [r, inf, sup, r_pct, inf_pct, sup_pct]
                        y_values += [r, inf, sup]
                        y_values_pct += [r_pct, inf_pct, sup_pct]
                    else:
                        values += [0, 0, 0, 0, 0, 0]
                results_out.write("\t".join([str(x) for x in values]) + "\n")
                index += 1
                start_date += delta

        print("\n\nMax real: ", max(y_values), max(y_values_pct))
        print("Max cases: ", max(cases), max(cases_pct))
        print("\n\n")

        print(
            "Created LaTeX table data in",
            os.path.join(self.simulation_output, table_name),
        )
        print(
            "Use code below in Tikz, and put LaTeX table data in data/ directory:\n\n"
        )

        colors = [
            "red",
            "teal",
            "purple",
            "violet",
            "yellow",
            "pink",
            "olive",
            "orange",
            "cyan",
            "green",
        ]

        print(
            """%%%%%%%%%%%%%%%%%%%%%
    Draw the confidence intervals
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
        )
        for i, name in enumerate(experiments):
            print(f"%{name}")
            print(
                f"\\addplot [stack plots=y, fill=none, draw=none, forget plot]   table [x=x, y={'-'.join(map(str, name))}_pct__inf, col sep=tab]   {{data/{table_name}}} \closedcycle;"
            )
            print(
                f"\\addplot [stack plots=y, fill={colors[i % len(colors)]}, fill opacity=0.15, draw opacity=0, forget plot]   table [x=x, y expr=\\thisrow{{{'-'.join(map(str, name))}_pct__sup}}-\\thisrow{{{'-'.join(map(str, name))}_pct__inf}}, col sep=tab]   {{data/{table_name}}} \closedcycle;"
            )
            print(
                f"\\addplot [stack plots=y, stack dir=minus, forget plot, draw=none] table [x=x, y={'-'.join(map(str, name))}_pct__sup] {{data/{table_name}}};"
            )

        print("\n\n% Draw the lines themselves")

        print(
            f"\\addplot [color=blue, mark=*] table[x=x, y=cases_times_{scale_factor}_pct, col sep=tab, legend] {{data/{table_name}}};"
        )
        for i, name in enumerate(experiments):
            print(
                f"\\addplot [color={colors[i % len(colors)]}, mark=*] table[x=x, y={'-'.join(map(str, name))}_pct, col sep=tab, legend] {{data/{table_name}}};"
            )


if __name__ == "__main__":
    start_plot()
