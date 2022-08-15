"""
If we run a simulation where one norm or EO is missing,
can be say something about the effect of each individual norm/EO?
"""
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

GroupedSimulations = Dict[str, Dict[Tuple[str], List[int]]]


def load_infected() -> GroupedSimulations:
    input_dir = sys.argv[1]

    grouped_simulations = defaultdict(lambda: defaultdict(list))
    for sim in os.listdir(input_dir):
        match = re.findall(r'experiment-without-(\w+)(?:-([\w-]+))?-run\d+', sim)
        if match and "epicurve.sim2apl.csv" in os.listdir(os.path.join(input_dir, sim)):
            name, params = match[0]
            params = [
                x
                    .replace("-for-", ",")
                    .replace("HIGHER_EDUCATION", "HE")
                    .replace("-over-", ">=")
                    .replace("7DMVoffices", "DMV")
                for x in params.split("-and-")
            ]
            infected = sum(
                map(int, open(
                    os.path.join(os.path.join(input_dir, sim), "epicurve.sim2apl.csv")
                ).readlines()[-1][:-1].split(";")[-4:])
            )
            grouped_simulations[name][tuple(params)].append(infected)

    return grouped_simulations


def group_and_sort(simulations: GroupedSimulations) -> List[Tuple[str, List[Tuple[str]]]]:
    grouped_and_sorted = list()
    sorted_keys = sorted(simulations, key=lambda x: max(map(max, simulations[x].values())))
    for key in sorted_keys:
        sublist = sorted(simulations[key].items(), key=lambda x: max(x[1]))
        grouped_and_sorted.append((key, sublist))
    return grouped_and_sorted


def plot_infected():
    grouped_and_sorted = group_and_sort(load_infected())
    values = [x[1] for y in grouped_and_sorted for x in y[1]]
    names = [" + ".join(x[0]) for y in grouped_and_sorted for x in y[1]]

    fig, ax = plt.subplots(figsize=[12.8, 15])
    x = np.arange(len(names))

    ax.bar(x, [np.average(y_) for y_ in values], yerr=[np.std(y_) for y_ in values], capsize=3)
    ax.set_xticks(x, labels=names, rotation=90, ha='center')  # , fontsize=8, color="gray")

    delta = x[0] + x[1] / 2  # Should just be 0.5

    major_tick_index = 0
    for name, values_for_name in grouped_and_sorted:
        ax.text(
            major_tick_index + len(values_for_name) / 2 - delta,
            -.0055,
            name,
            weight="bold",
            ha='center',
            va='top',
            rotation=90,
            transform=ax.get_xaxis_transform()
        )
        major_tick_index += len(values_for_name)
        line = Line2D(
            [major_tick_index-delta, major_tick_index-delta],
            [0.01, -.15],
            color="gray",
            transform=ax.get_xaxis_transform()
        )
        line.set_clip_on(False)
        ax.add_line(line)

    ax.set_ylim(min(map(min, values)), max(map(max, values)))
    ax.yaxis.grid(True)
    ax.set_ylabel("# infected")

    plt.title("Impact per norm on infections")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_infected()
