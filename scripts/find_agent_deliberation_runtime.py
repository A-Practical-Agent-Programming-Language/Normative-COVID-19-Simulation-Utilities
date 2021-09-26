import math
import os
import re
from collections import defaultdict
from typing import Tuple, List, Dict

import numpy as np

from timings_rough_analysis import split_norms

import click


@click.command()
@click.option(
    "--simulation-output",
    "-s",
    help="Directory containing output of one simulation, or containing multiple "
         "simulation configurations (if so, all simulations in directory will be plotted",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
    required=True,
)
def start(simulation_output):
    files = list_files(simulation_output)
    per_agent_cummulative, per_cycle_cummulative = defaultdict(list), defaultdict(list)
    for directory in files:
        norm_count_by_time_step, time_step_by_norm_count = find_norms_by_time_step(directory)
        deliberation_duration_by_time_step = read_deliberation_duration(directory, norm_count_by_time_step)
        per_agent, per_cycle = group_time_by_norms(deliberation_duration_by_time_step)
        for norm in per_agent:
            per_agent_cummulative[norm] += per_agent[norm]
            per_cycle_cummulative[norm] += per_cycle[norm]

    print("Average time per agent:")
    print_timing_by_norm(per_agent_cummulative)
    print_timing_overall(per_agent_cummulative)
    print("Average time per deliberation cycle:")
    print_timing_by_norm(per_cycle_cummulative)
    print_timing_overall(per_cycle_cummulative)


def list_files(directory: str) -> List[Tuple[str, str]]:
    files = list()
    for sub_directory in os.listdir(directory):
        sub_directory = os.path.join(directory, sub_directory)
        if os.path.isdir(sub_directory):
            average_schedules = None
            run_log = None
            for x in os.listdir(sub_directory):
                if x.startswith("average-schedules"):
                    average_schedules = os.path.join(sub_directory, x)
                elif x.startswith("calibration.agents"):
                    run_log = os.path.join(sub_directory, x)
                if average_schedules is not None and run_log is not None:
                    break
            if average_schedules is not None and run_log is not None:
                files.append((average_schedules, run_log))

    return files


def find_norms_by_time_step(directory: Tuple[str, str]) -> (List[Tuple[int, str]], Dict[int, int]):
    """
    Find how many norms are active at each time step
    Args:
        directory:

    Returns:

    """
    active_norms = []
    norm_count_by_time_step = list()
    time_step_by_norm_count = defaultdict(list)
    with open(directory[0], 'r') as average_schedule_in:
        headers = average_schedule_in.readline()[:-1].split(";")
        for line in average_schedule_in:
            line = line[:-1].split(";")
            norms_in = split_norms(line[headers.index("NORMS_ACTIVATED")])
            norms_out = split_norms(line[headers.index("NORMS_DEACTIVATED")])
            for norm in norms_in:
                if norm not in active_norms:
                    active_norms.append(norm)
            for norm in norms_out:
                if norm in active_norms:
                    active_norms.remove(norm)
            time_step_by_norm_count[len(active_norms)].append(len(norm_count_by_time_step))
            norm_count_by_time_step.append((len(active_norms), line[headers.index("Date")]))

    return norm_count_by_time_step, time_step_by_norm_count


def read_deliberation_duration(directory: Tuple[str, str], norm_count_by_time_step: List[Tuple[int, str]]) -> List[Dict[str, int or float]]:
    """
    Find the time it took to execute one deliberation cycle, and the number of agents this counted for
    Args:
        directory:

    Returns:

    """
    deliberation_duration_by_time_step = list()
    regex = re.compile(r'Tick (\d+) took (\d+) milliseconds for (\d+) agents \(roughly (0.\d+)ms per agent\)')
    with open(directory[1], 'r') as run_log_in:
        for line in run_log_in:
            match = regex.findall(line)
            if match:
                time_step = len(deliberation_duration_by_time_step)
                assert int(match[0][0]) == time_step
                values = dict(zip(["tick", "millis", "n_agents", "per_agent_millis"], list(map(lambda x: (int(x[0]), int(x[1]), int(x[2]), float(x[3])), match))[0]))
                re_calculated = values["millis"] / values["n_agents"]
                assert math.fabs(re_calculated - values["per_agent_millis"]) < 10e-7
                values["per_agent_millis"] = re_calculated / 8
                values["norms"] = norm_count_by_time_step[time_step][0]
                deliberation_duration_by_time_step.append(values)

    return deliberation_duration_by_time_step


def group_time_by_norms(deliberation_durations: List[Dict[str, int or float]]) -> (Dict[int, List[float]], Dict[int, List[int]]):
    per_agent_by_norm = defaultdict(list)
    per_cycle_by_norm = defaultdict(list)
    for duration in deliberation_durations:
        per_agent_by_norm[duration["norms"]].append(duration["per_agent_millis"])
        per_cycle_by_norm[duration["norms"]].append(duration["millis"])
    return per_agent_by_norm, per_cycle_by_norm


def print_timing_by_norm(time_by_norms: Dict[int, List[int or float]]):
    for norm in sorted(time_by_norms.keys()):
        print(f"\t{norm} norms: average {np.average(time_by_norms[norm])} ± {np.std(time_by_norms[norm])} milliseconds   ({len(time_by_norms[norm])} samples)")


def print_timing_overall(time_by_norms: Dict[int, List[int or float]]):
    timings = list()
    for by_norm in time_by_norms.values():
        timings += by_norm

    print(f"\tAll norms combined: {np.average(timings)} ± {np.std(timings)} milliseconds   ({len(timings)} samples)")


if __name__ == "__main__":
    start()
