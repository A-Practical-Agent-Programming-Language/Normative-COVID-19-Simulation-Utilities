import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from utility.plots import add_norms_to_graph

runs = os.path.join("..", "output", "agent-runs", "trust-discount")
run_re = r'trust-discount-factor-([0-9.]+)-sample-random-(\w+)-mode-liberal-([0-9.]+)-mode-conservative-([0-9.]+)-run(\d+)'
counties = os.listdir(runs)


def plot_in_dir(base, dir):
    grouped_runs = group_runs(base, dir)
    for sample_random, modes in grouped_runs.items():
        sample_random = sample_random == "True"
        for ((mode_liberal, mode_conservative), learning_rates) in modes.items():
            mode_liberal = float(mode_liberal)
            mode_conservative = float(mode_conservative)
            combined_data = dict()
            for learning_rate, simulations in learning_rates.items():
                curve, trust = load_data_for_run(simulations)
                combined_data[learning_rate] = curve
                plot_one_experiment(sample_random, mode_liberal, mode_conservative, learning_rate, curve, trust, simulations)
            plot_merged_experiments(sample_random, mode_liberal, mode_conservative, simulations['0'], combined_data)


def plot_merged_experiments(sample_random: bool, lib: float, cons: float, directory: str, data: Dict[float, Dict[int, Dict[str, float]]], plot_rolling_average=True):
    fig, ax = plt.subplots()
    for learning_rate in sorted(list(data.keys())):
        curve = data[learning_rate]
        for fips, gyration_data in curve.items():
            dates = sorted(list(gyration_data.keys()))
            values = [gyration_data[x] for x in dates]
            if plot_rolling_average:
                values = gyration_rolling_average(values)
            ax.plot(dates, values, label=f"η = {learning_rate}")

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
    plt.close()


def plot_one_experiment(
        sample_random: bool,
        lib: float,
        cons: float,
        learning_rate: float,
        gyration: Dict[int, Dict[str, float]],
        trust: Dict[int, Dict[str, float]],
        directories,
        plot_rolling_average = True
):
    fig, ax = plt.subplots()

    dates = sorted(list(gyration[list(gyration.keys())[0]].keys()))
    plt.xticks(np.arange(0, len(dates), 1), map(lambda x: x[5:], dates), rotation=60)
    ax.xaxis.set_major_locator(MultipleLocator(7))
    ax.xaxis.set_minor_locator(MultipleLocator(3.5))
    ax2 = ax.twinx()
    ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
    ax2.set_ylim([0,1])
    for fips, data in gyration.items():
        values = [data[x] for x in dates]
        if plot_rolling_average:
            values = gyration_rolling_average(values)
        trust_values = [trust[fips][x] for x in dates]

        ax.plot(dates, values, label=f"Gyration {fips}")
        ax2.plot(dates, trust_values, label=f"Trust {fips}")

    add_norms_to_graph(
        ax, dates=dates, simulation_output_dir=directories['0']
    )

    title = f"Learning rate η = {learning_rate}"
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
    plt.close()


def gyration_rolling_average(values: List[float], window_size: int = 7):
    new_values = list()
    for i in range(len(values)):
        if i < window_size:
            new_values.append(float('nan'))
        else:
            window = values[i-window_size:i]
            new_values.append(sum(window) / window_size)
    return new_values


def load_data_for_run(run: Dict[int, str]):
    mobility = list()
    for run_number, directory in run.items():
        data = defaultdict(lambda: defaultdict(dict))
        with open(os.path.join(directory, 'tick-averages.csv'), 'r') as run_in:
            run_in.readline()
            for line in run_in:
                date, fips, gyration, trust, agents = line[:-1].split(",")
                data[date][fips]['gyration'] = float(gyration) * int(agents)
                data[date][fips]['agents'] = int(agents)
                data[date][fips]['trust'] = float(trust) * int(agents)
        mobility.append(data)

    merged = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data in mobility:
        for date, values in data.items():
            for fips, information in values.items():
                merged[date][fips]['gyration'] += information['gyration']
                merged[date][fips]['agents'] += information['agents']
                merged[date][fips]['trust'] += information['trust']

    final_merged = defaultdict(dict)
    final_merged_trust = defaultdict(dict)
    for date, values in merged.items():
        for fips, information in values.items():
            final_merged[fips][date] = information['gyration'] / information['agents']
            final_merged_trust[fips][date] = information['trust'] / information['agents']

    return final_merged, final_merged_trust


def group_runs(base, dir):
    grouped_runs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for run in os.listdir(os.path.join(base, dir)):
        match = re.findall(run_re, run)
        if match:
            learning_rate, sample_random, mode_liberal, mode_conservative, n_run = match[0]
            grouped_runs[sample_random][(mode_liberal, mode_conservative)][learning_rate][n_run] = os.path.join(base, dir, run)

    return grouped_runs


if __name__ == "__main__":
    for county in counties:
        if os.path.isdir(os.path.join(runs, county)):
            plot_in_dir(runs, county)
