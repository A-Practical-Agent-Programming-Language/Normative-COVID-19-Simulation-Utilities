import os
import re
from collections import defaultdict
from typing import List, Dict

import click as click
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from scripts.plot_epicurve_average import EpicurvePlotter
from utility.plots import metrics

GROUPED_RUN = Dict[int, Dict[int, Dict[str, str]]]
GROUPED_EPICURVE = Dict[int, Dict[int, list]]

class PlotExperiment(object):

    def __init__(self, exp_output):
        self.exp_output = exp_output
        self.grouped_runs = self.group_runs()
        self.grouped_epicurves, self.population_size = self.read_epicurves()
        for curve in sorted(self.grouped_epicurves):
            self.plot_curves(
                curve,
                self.grouped_runs[curve][0]["until"],
                self.merge_curves(self.grouped_epicurves[curve])
            )

    def group_runs(self) -> GROUPED_RUN:
        grouped_runs = defaultdict(dict)
        for directory in os.listdir(self.exp_output):
            if os.path.exists(os.path.join(self.exp_output, directory, "epicurve.sim2apl.csv")):
                match = re.findall(r'^experiment-(\d+)-norms-until(\d{4}-\d{2}-\d{2})-run(\d+)$', directory)
                if len(match):
                    experiment, until_date, run = match[0]
                    grouped_runs[int(experiment)][int(run)] = dict(
                        until=until_date,
                        directory=os.path.join(self.exp_output, directory)
                    )

        return grouped_runs

    def read_epicurves(self):
        epicurves = defaultdict(list)
        population_size = 0
        for experiment in self.grouped_runs:
            for run, run_values in self.grouped_runs[experiment].items():
                epicurve, population_size = EpicurvePlotter.read_epicurve_file(run_values['directory'])
                epicurves[experiment].append(epicurve)

        return epicurves, population_size

    def merge_curves(self, curves):
        keys = ['NOT_SET', 'SUSCEPTIBLE', 'EXPOSED', 'INFECTED_SYMPTOMATIC', 'INFECTED_ASYMPTOMATIC', 'RECOVERED']
        merged = defaultdict(lambda: defaultdict(list))

        for curve in curves:
            for date in curve:
                for key in keys:
                    merged[date][key].append(curve[date][key])
                    merged[date]['INF_TOTAL'].append(curve[date]["EXPOSED"] + curve[date]["INFECTED_SYMPTOMATIC"] + curve[date]["INFECTED_ASYMPTOMATIC"])

        return merged

    def plot_curves(self, experiment: int, date: str, curves: Dict[str, Dict[str, List[int]]]):
        dates = sorted(list(curves.keys()))
        x = np.arange(0, len(dates), 1)
        fig, ax = plt.subplots()

        for key, label, color in [("SUSCEPTIBLE", "Susceptible", '#004586'), ("INF_TOTAL", "Infected", '#ff420e'), ("RECOVERED", "Recovered", '#ffd320')]:
            values, error = metrics([curves[date][key] for date in curves])
            color = ax.plot(x, values, label=label, color=color)[0].get_color()
            ax.fill_between(
                x,
                values - error,
                values + error,
                alpha=0.2,
                facecolor=color,
                antialiased=True
            )

        plt.legend()
        plt.suptitle(f"First {experiment} EOs")
        plt.title(f"All interventions up to and including {date}")
        plt.xticks(x, map(lambda y: y[5:], dates), rotation=90)
        ax.xaxis.set_major_locator(MultipleLocator(7))
        ax.set_ylabel("Number of agents")
        ax.set_xlabel("Simulation day (month-day in 2020)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_output, f"experiment-{experiment}-norms-until{date}.png"), dpi=300)
        plt.show()


@click.command()
@click.option(
    "--exp-output",
    "-e",
    help="Directory containing all experiment output runs",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, resolve_path=True),
    required=True
)
def plot_experiment(exp_output):
    pe = PlotExperiment(exp_output)


if __name__ == "__main__":
    plot_experiment()
