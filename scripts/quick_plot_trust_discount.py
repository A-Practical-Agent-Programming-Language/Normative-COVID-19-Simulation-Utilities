import os
import re
from collections import defaultdict
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from scripts.plot_epicurve_average import EpicurvePlotter
from scripts.plot_mobility_average import MobilityPlotter
from utility.plots import add_norms_to_graph

runs = os.path.join("..", "output", "agent-runs", "trust-discount")
run_re = r'trust-discount-factor-([0-9.]+)-sample-random-(\w+)-mode-liberal-([0-9.]+)-mode-conservative-([0-9.]+)-run(\d+)'
counties = os.listdir(runs)


def plot_in_dir(base, dir):
    grouped_runs = group_runs(base, dir)
    for sample_random, modes in grouped_runs.items():
        sample_random = sample_random == "True"
        for ((mode_liberal, mode_conservative), discount_factors) in modes.items():
            mode_liberal = float(mode_liberal)
            mode_conservative = float(mode_conservative)
            combined_data = dict()
            for discount_factor, simulations in discount_factors.items():
                curve = load_data_for_run(simulations)
                combined_data[discount_factor] = curve
                plot_one_experiment(sample_random, mode_liberal, mode_conservative, discount_factor, curve, simulations)
            plot_merged_experiments(sample_random, mode_liberal, mode_conservative, simulations['0'], combined_data)


def plot_merged_experiments(sample_random: bool, lib: float, cons: float, directory: str, data: Dict[float, Dict[int, Dict[str, float]]]):
    fig, ax = plt.subplots()
    for discount_factor, curve in data.items():
        for fips, gyration_data in curve.items():
            dates = sorted(list(gyration_data.keys()))
            values = [gyration_data[x] for x in dates]

            ax.plot(dates, values, label=discount_factor)

    add_norms_to_graph(
        ax, dates=dates, simulation_output_dir=directory
    )
    plt.xticks(np.arange(0, len(dates), 1), map(lambda x: x[5:], dates), rotation=60)
    ax.xaxis.set_major_locator(MultipleLocator(7))
    ax.xaxis.set_minor_locator(MultipleLocator(3.5))

    if sample_random:
        title = "Uniformly sampled trust"
    else:
        title = f"Mode liberal: {lib:.3f} -- mode conservative: {cons:.3f}"

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(runs, title.replace(" ", "-").replace(":", "-") + ".png"), dpi=300)

def plot_one_experiment(sample_random: bool, lib: float, cons: float, discount_factor: float, gyration: Dict[int, Dict[str, float]], directories):
    fig, ax = plt.subplots()
    for fips, data in gyration.items():
        dates = sorted(list(data.keys()))
        values = [data[x] for x in dates]

        ax.plot(dates, values)

    add_norms_to_graph(
        ax, dates=dates, simulation_output_dir=directories['0']
    )

    plt.xticks(np.arange(0, len(dates), 1), map(lambda x: x[5:], dates), rotation=60)
    ax.xaxis.set_major_locator(MultipleLocator(7))
    ax.xaxis.set_minor_locator(MultipleLocator(3.5))

    title = f"Discount factor {discount_factor}"
    if sample_random:
        subtitle = "Uniformly sampled trust"
    else:
        subtitle = f"Mode liberal: {lib:.3f} -- mode conservative: {cons:.3f}"

    plt.title(subtitle)
    plt.suptitle(title)
    plt.legend()

    out_dir = os.path.join(runs, subtitle.replace(" ", "-").replace(":", "-"))

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, title.replace(" ", "-") + ".png"), dpi=300)


def load_data_for_run(run: Dict[int, str]):
    mobility = list()
    for run_number, directory in run.items():
        data = defaultdict(lambda: defaultdict(dict))
        with open(os.path.join(directory, 'tick-averages.csv'), 'r') as run_in:
            run_in.readline()
            for line in run_in:
                date, fips, gyration, agents = line[:-1].split(",")
                data[date][fips]['gyration'] = float(gyration) * int(agents)
                data[date][fips]['agents'] = int(agents)
        mobility.append(data)

    merged = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data in mobility:
        for date, values in data.items():
            for fips, information in values.items():
                merged[date][fips]['gyration'] += information['gyration']
                merged[date][fips]['agents'] += information['agents']

    final_merged = defaultdict(dict)
    for date, values in merged.items():
        for fips, information in values.items():
            final_merged[fips][date] = information['gyration'] / information['agents']

    return final_merged


def group_runs(base, dir):
    grouped_runs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for run in os.listdir(os.path.join(base, dir)):
        match = re.findall(run_re, run)
        if match:
            discount_factor, sample_random, mode_liberal, mode_conservative, n_run = match[0]
            grouped_runs[sample_random][(mode_liberal, mode_conservative)][discount_factor][n_run] = os.path.join(base, dir, run)

    return grouped_runs



if __name__ == "__main__":
    for county in counties:
        if os.path.isdir(os.path.join(runs, county)):
            plot_in_dir(runs, county)
