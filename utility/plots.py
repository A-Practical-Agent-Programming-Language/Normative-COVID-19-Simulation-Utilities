import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines
from scipy import stats as st


def metrics(measure: List[List[float]]) -> (List[float], List[float]):
    """
    Given a list of lists of values, calculates the mean and standard deviation of each list
    in metrics
    Args:
        measure: List of lists of values on which to calculate mean and average

    Returns:
        Two lists of values, the first containing the mean, and the second the standard deviation,
        calculated from metrics
    """
    return np.array(list(map(lambda x: np.mean(x), measure))), np.array(
        list(map(lambda x: np.std(x), measure))
    )


def ci_metrics(measure: List[List[float]], interval=0.95) -> (List[float], List[float], List[float]):
    mean = np.array(list(map(lambda x: np.mean(x), measure)))
    ci = np.array(list(map(lambda x: st.t.interval(interval, len(x)-1, loc=np.mean(x), scale=st.sem(x)), measure)))
    lower = [x[0] for x in ci]
    upper = [x[1] for x in ci]

    return mean, lower, upper


def add_norms_to_graph(
    ax: plt.Axes,
    dates: List[str],
    norms: (List[str], List[str]) = None,
    simulation_output_dir: str = None,
    norm_schedule_file: str = None,
) -> None:
    """
    Given the subAxes of a graph with dates on the x-axes, draw red and green lines corresponding to norm activation
    and deactivation respectively.

    Args:
        ax:     Plot subaxes
        dates:  List of dates used on x-axes
        norms:  Tuple of list of dates; first element for activated norms, second for deactivated
        simulation_output_dir: A simulation output directory that contains an average-schedule csv file
        norm_schedule_file: The norm schedule as used by the simulation

        Only one of the (*norms*, *simulation_output_dir*, and *norm_schedule_file*) arguments needs to
        be given
    """
    if norms is None:
        if simulation_output_dir is not None:
            activated, deactivated = read_norms_from_average_schedule(
                simulation_output_dir
            )
        elif norm_schedule_file is not None:
            activated, deactivated = read_norms_dates_from_norm_schedule(norm_schedule_file)
        else:
            raise ValueError(
                "No norms provided, can't add norms if we don't have them!"
            )
    else:
        activated, deactivated = norms

    lines_to_draw = list()
    for norm in activated:
        if norm in dates:
            lines_to_draw.append((dates.index(norm), "red"))
    for norm in deactivated:
        if norm in dates:
            lines_to_draw.append((dates.index(norm), "green"))

    for (x_pos, color) in lines_to_draw:
        line = lines.Line2D(
            [x_pos, x_pos],
            [ax.dataLim.ymin, ax.dataLim.ymax],
            color=color,
            linewidth=1,
            linestyle=":",
        )
        ax.add_line(line)


def read_norms_dates_from_norm_schedule(norm_schedule_file: str) -> (List[str], List[str]):
    """
    This function generates two lists of dates. The first list are all the dates on which a new norm is being
    activated (with duplicates if multiple norms are activated) while the second list contains all dates on which
    norms are being deactivated

    Args:
        norm_schedule_file:

    Returns:

    """
    new_norm_dates, norm_deactivated_dates = list(), list()
    with open(norm_schedule_file, "r") as norms_in:
        norms_in.readline()[:-1].split(",")  # Skip header
        for line in norms_in:
            line = line[:-1].split(",")
            if line[0].strip(" ") != "":
                new_norm_dates.append(line[0])
            if line[2].strip(" ") != "":
                norm_deactivated_dates.append(line[2])
    return new_norm_dates, norm_deactivated_dates


def read_norms_from_average_schedule(
    simulation_output_directory: str,
) -> (List[str], List[str]):
    """
        This function generates two lists of dates. The first list are all the dates on which a new norm is being
        activated (with duplicates if multiple norms are activated) while the second list contains all dates on which
        norms are being deactivated

        Args:
            norm_schedule_file:

        Returns:

        """
    average_schedule_file = next(
        filter(
            lambda x: "average-schedule" in x, os.listdir(simulation_output_directory)
        )
    )
    new_norm_dates, norm_deactivated_dates = list(), list()
    with open(
        os.path.join(simulation_output_directory, average_schedule_file), "r"
    ) as norms_in:
        norms_in.readline()  # Skip header
        for line in norms_in:
            date, _, activated, deactivated = line[:-1].split(";")[:4]
            if len(activated.strip(" ")):
                new_norm_dates += [date] * len(activated.split(","))
            if len(deactivated.strip(" ")):
                norm_deactivated_dates += [date] * len(deactivated.split(","))

    return new_norm_dates, norm_deactivated_dates

