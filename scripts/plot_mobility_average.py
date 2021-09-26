import datetime
import math
import os
from collections import defaultdict
from typing import Dict, List

import click
import numpy as np
from matplotlib import pyplot as plt, lines
from matplotlib.ticker import MultipleLocator

from classes.Gyration import Gyration
from utility.plots import metrics, add_norms_to_graph
from utility.run_finder import find_behavior_runs, runs_to_rmse_list
from utility.utility import load_toml_configuration

Date = str  # Date formatted as YYYY-MM-DD


class MobilityPlotter(object):
    """
    Can plot the change in mobility for one or more simulation runs (with the same parameter configuration)
    into a single plot, together with the activation and deactivations of norms

    Can also produce a LaTeX table for use in publications

    See:
        plot_mobility_average.py --help
    """

    def __init__(
        self,
        simulation_output,
        target_file,
        county_configuration,
        mobility_index_file,
        sliding_window_size,
        print_table,
    ):
        self.simulation_output = simulation_output
        self.counties = load_toml_configuration(county_configuration)
        self.fips_to_county = self.make_fips_to_county_name_dict()
        self.g = Gyration(
            mobility_index_file,
            "tick-averages.csv",
            self.counties["counties"],
            sliding_window_size,
        )
        configurations = find_behavior_runs(simulation_output)

        # If a large directory is provided with more than one run configuration, plot each run configuration separately
        for configuration, run_group in configurations.items():
            # Ensure target file is present
            rmse = self.g.calculate_rmse(runs_to_rmse_list(run_group))
            print("RMSE", rmse, configuration)

            mobilities = list()
            for run in run_group:
                mobilities.append(
                    self.read_mobiltiy_file(os.path.join(run, target_file))
                )
            merged = self.merge_mobility(mobilities)

            file_name = "{rmse:.4f}rmse-{0:.4f}l-{1:.4f}r-{2:.4f}f-{3:.0f}fs-combined.png".format(
                *configuration, rmse=rmse
            )
            self.plot_merged_mobility(run_group[0], merged, name=file_name)

            if print_table:
                self.create_data_table(merged, rmse)

            for fips in merged[list(merged.keys())[0]]:
                fname = "{rmse:.4f}rmse-{0:.4f}l-{1:.4f}r-{2:.4f}f-{3:.0f}fs-{county}.png".format(
                    *configuration,
                    rmse=rmse,
                    county=self.get_county_name_for_fips(fips),
                )
                self.plot_merged_mobility(run_group[0], merged, fips, fname)

    def make_fips_to_county_name_dict(self) -> Dict[int, str]:
        fips_to_county_name = dict()
        for name, county in self.counties["counties"].items():
            fips_to_county_name[county["fipscode"]] = name
        return fips_to_county_name

    def get_county_name_for_fips(self, fips: int or str) -> str:
        fips = int(fips)
        if fips in self.fips_to_county:
            return self.fips_to_county[fips]

        return str(fips)

    @staticmethod
    def read_mobiltiy_file(
        mobility_file: str,
    ) -> Dict[Date, Dict[int, Dict[str, float]]]:
        mobility = defaultdict(dict)

        with open(mobility_file, "r") as mob_in:
            headers = mob_in.readline()[:-1].split(",")
            subheaders = mob_in.readline()[:-1].split(",")
            fipses = list(set(headers[1:]))
            headers = list(zip(headers, subheaders))

            for line in mob_in:
                mob = dict(zip(headers, line.split(",")[:-1]))
                for fips in fipses:
                    mobility[mob["date", ""]][fips] = dict(
                        real=mob[fips, "real"], predicted=mob[fips, "agents"]
                    )

        return mobility

    @staticmethod
    def merge_mobility(
        mobilities: List[Dict[Date, Dict[int, Dict[str, float]]]]
    ) -> Dict[Date, Dict[int, Dict[str, List[float]]]]:
        merged = dict()
        for mobility in mobilities:
            for day in mobility:
                if day not in merged:
                    merged[day] = dict()
                for fips in mobility[day]:
                    if fips not in merged[day]:
                        merged[day][fips] = dict(real=[], predicted=[])
                    r = mobility[day][fips]["real"]
                    p = mobility[day][fips]["predicted"]
                    merged[day][fips]["real"].append(
                        float("nan") if r == "" else float(r)
                    )
                    merged[day][fips]["predicted"].append(
                        float("nan") if p == "" else float(p)
                    )

        return merged

    def plot_merged_mobility(
        self,
        sample_simulation_dir: str,
        mobility,
        fips_to_plot=None,
        name="mobility.png",
    ):
        fig, ax = plt.subplots()
        days = sorted(list(set(mobility.keys())))
        x = np.arange(0, len(days), 1)

        for fips in mobility[list(mobility.keys())[0]]:
            if fips_to_plot is not None and fips is not fips_to_plot:
                continue

            real = [mobility[day][fips]["real"][0] for day in mobility]
            agents = [mobility[day][fips]["predicted"] for day in mobility]

            county = self.get_county_name_for_fips(fips)
            real_label = f"Cuebiq {county}" if fips_to_plot is None else "Cuebiq"
            agents_label = f"agents {county}" if fips_to_plot is None else "Agents"

            y, err = metrics(agents)
            ax.plot(x, real, label=real_label, linestyle="--")
            line = ax.plot(x, y, label=agents_label)
            c = line[0].get_color()
            ax.fill_between(
                x, y - err, y + err, alpha=0.2, facecolor=c, antialiased=True
            )

        add_norms_to_graph(ax, days, simulation_output_dir=sample_simulation_dir)

        plt.suptitle("Mobility")
        plt.title(name[:-4])

        plt.xticks(x, map(lambda x: x[5:], days), rotation=90)
        ax.xaxis.set_major_locator(MultipleLocator(7))
        ax.xaxis.set_minor_locator(MultipleLocator(3.5))
        plt.xlabel("Simulation day")
        plt.ylabel("Percentage change in mobility")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.simulation_output, name), dpi=300)
        plt.show()

    def create_data_table(self, mobility, rmse: float):
        """
        Creates a tex file with all the data points, and prints the code that can be interpreted by Tikz (in LaTeX)
        to plot that data
        """
        start_date = datetime.date(2020, 1, 1)
        end_date = datetime.date(2020, 12, 1)
        delta = datetime.timedelta(days=1)

        fips = sorted(list(set(mobility[list(mobility.keys())[0]].keys())))
        headers = ["x", "date"]
        for f in fips:
            headers += [f"{f}_real", f, f"{f}__inf", f"{f}__sup"]

        all_values = list()

        table_name = f"mobility_calibration_rmse{rmse}.tex"

        with open(os.path.join(self.simulation_output, table_name), "w") as result_out:
            result_out.write("\t".join(headers) + "\n")
            index = 0
            while start_date < end_date:
                values = [index, start_date]
                for f in fips:
                    if str(start_date) in mobility and f in mobility[str(start_date)]:
                        p = mobility[str(start_date)][f]["predicted"]
                        real = mobility[str(start_date)][f]["real"][0]
                        y = np.mean(p)
                        err = np.std(p)
                        values += [real, y, y + err, y - err]
                        all_values += [
                            x for x in [real, y, y + err, y - err] if not math.isnan(x)
                        ]
                    else:
                        values += ["nan", "nan", "nan", "nan"]

                result_out.write("\t".join([str(x) for x in values]) + "\n")
                index += 1
                start_date += delta

        print(
            "Created LaTeX table data in",
            os.path.join(self.simulation_output, table_name),
        )
        print(
            "Use code below in Tikz, and put LaTeX table data in data/ directory:\n\n"
        )

        colors = [
            "teal",
            "cyan",
            "orange",
            "purple",
            "violet",
            "yellow",
            "olive",
            "pink",
            "green",
            "red",
        ]

        print(f"Max: {np.max(all_values)}")
        print(f"Min: {np.min(all_values)}")
        print("\n\n")

        print(
            """%%%%%%%%%%%%%%%%%%%%%
    Draw the confidence intervals
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
        )
        for i, name in enumerate(fips):
            print(f"%{name}")
            print(
                f"\\addplot [stack plots=y, fill=none, draw=none, forget plot]   table [x=x, y={name}__inf, col sep=tab]   {{data/{table_name}}} \closedcycle;"
            )
            print(
                f"\\addplot [stack plots=y, fill={colors[i % len(colors)]}, fill opacity=0.15, draw opacity=0, forget plot]   table [x=x, y expr=\\thisrow{{{name}__sup}}-\\thisrow{{{name}__inf}}, col sep=tab]   {{data/{table_name}}} \closedcycle;"
            )
            print(
                f"\\addplot [stack plots=y, stack dir=minus, forget plot, draw=none] table [x=x, y={name}__sup] {{data/{table_name}}};"
            )

        print("\n\n% Draw the lines themselves")

        for i, name in enumerate(fips):
            print(
                f"\\addplot [color={colors[i % len(colors)]}, mark=*, mark size=.3, densely dashed, forget plot] table[x=x, y={name}_real, col sep=tab, legend] {{data/{table_name}}};"
            )
            print(
                f"\\addplot [color={colors[i % len(colors)]}, mark=star, mark size=0.7, forget plot] table[x=x, y={name}, col sep=tab, legend] {{data/{table_name}}};"
            )

        legend = ["Cuebiq", "Simulation"]
        print("%legend")
        print(
            "\\addplot +[mark=*, mark size=.3, densely dashed, color=black] coordinates {(0,0) (0,0)};"
        )
        print(
            "\\addplot +[mark=star, mark size=0.7, color=black] coordinates {(0,0) (0,0)};"
        )

        for i, name in enumerate(fips):
            legend.append(self.get_county_name_for_fips(name))
            print(
                f"\\addplot +[mark=star, mark size=0.0, color={colors[i % len(colors)]}] coordinates {{(0,0) (0,0)}};"
            )

        print("\n")
        legend = [f"\\tiny{{{x}}}" for x in legend]
        print("\\legend{" + ", ".join(legend) + "}")


@click.command()
@click.option(
    "--simulation-output",
    "-s",
    help="Directory containing output of one simulation, or containing multiple "
    "simulation configurations (if so, all simulations in directory will be plotted",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
    required=True,
)
@click.option(
    "--target-file",
    "-f",
    help="Specify the file from which to read mobility data",
    type=str,
    required=False,
    default="mobility_index.csv",
)
@click.option(
    "--county-configuration",
    "-c",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
    help="Specify the county configuration used for this simulation run containing all simulated counties",
    required=True,
)
@click.option(
    "--mobility-index-file",
    "-m",
    type=click.Path(exists=True),
    help="Specify the location of the file containing the mobility index for each (relevant) county",
    default=os.path.join("..", "external", "va_county_mobility_index.csv"),
)
@click.option(
    "--sliding-window-size",
    type=int,
    default=7,
    help="Specify the size of the sliding window with which the mobility index will be smoothed when calculating the "
    "RMSE. Note that for the baseline, the actual size of the window used may be smaller, because some "
    "dates are missing",
)
@click.option(
    "--print-table",
    "-p",
    type=bool,
    default=False,
    help="If explicitly set to true, the script will create a file containing a table with the mobility data, "
    "and will print Tikz picture code to plot that data in LaTeX",
)
def start_plot(**args):
    MobilityPlotter(**args)


if __name__ == "__main__":
    start_plot()
