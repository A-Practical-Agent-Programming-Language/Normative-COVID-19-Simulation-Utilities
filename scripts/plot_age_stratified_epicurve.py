import os
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, PercentFormatter

from utility.utility import get_project_root


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
    "--case-estimations",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
    help="In order to plot the confidence interval of the estimated number of cases, provide the file that contains "
         "that data. It is provided by the COVID-19 projections project "
         "(https://covid19-projections.com/infections/us-va) and can be downloaded from "
         "https://raw.githubusercontent.com/youyanggu/covid19-infection-estimates-latest/main/counties/latest_VA.csv",
    default=None,
    required=False
)
@click.option(
    "--county-configuration",
    "-c",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
    help="Specify the county configuration used for this simulation run containing all simulated counties",
    required=True,
)
@click.option(
    "--standard-deviation",
    type=bool,
    default=True,
    help="If this flag is omitted or set to true, standard deviation will be used to plot the confidence interval,"
         "otherwise, the 95% confidence interval will be plotted"
)
def plot(simulation_output, real_cases, case_estimations, county_configuration, standard_deviation):
    if "epicurve.sim2apl.csv" in os.listdir(simulation_output):
        epicurve = [read_age_stratified_epicurve(simulation_output)]
        for epicurve, age_stratified_epicurves in epicurve:
            # TODO, make everything in these epicurves lists instead of single values, so other functions can be used
            age_stratified_epicurve_unit_test(epicurve, age_stratified_epicurves)
            plot_all_infected(age_stratified_epicurves, simulation_output)
        #     for title, age_stratified_epicurve in age_stratified_epicurves.items():
        #         plot_epicurve(age_stratified_epicurve, title)
    else:
        epicurves = list()
        for subdir in os.listdir(simulation_output):
            if os.path.isdir(os.path.join(simulation_output, subdir)) and "epicurve.sim2apl.csv" in os.listdir(os.path.join(simulation_output, subdir)):
                epicurves.append(read_age_stratified_epicurve(os.path.join(simulation_output, subdir)))

        epicurve, age_stratified_epicurves = merge_curves(epicurves)
        plot_all_infected(age_stratified_epicurves, simulation_output)


def merge_curves(epicurves):
    merged, merged_stratified = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for curves, stratified in epicurves:
        for date, vals in curves.items():
            for state, val in vals.items():
                merged[date][state].append(val)

        for age_group, epicurve in stratified.items():
            for date, vals in epicurve.items():
                for state, val in vals.items():
                    merged_stratified[age_group][date][state].append(val)

    return merged, merged_stratified


def plot_epicurve(epicurve, title):
    fig, ax = plt.subplots()
    x = sorted(list(epicurve.keys()))
    values = [
        ("Susceptible", [epicurve[date]["SUSCEPTIBLE"] for date in x]),
        ("Recovered", [epicurve[date]["RECOVERED"] for date in x]),
        ("Infected", [sum([epicurve[date][val] for val in ["EXPOSED", "INFECTED_SYMPTOMATIC", "INFECTED_ASYMPTOMATIC"]]) for date in x])
    ]
    total = sum([y[1][0] for y in values])
    print(total)

    pcts = list()
    for name, vals in values:
        pcts.append((name, [val / total * 100 for val in vals]))

    for name, pct in pcts:
        if name == "Infected":
            ax.plot(x, pct, label=name)

    plt.xticks(x, map(lambda y: y[5:], x), rotation=90)
    ax.xaxis.set_major_locator(MultipleLocator(7))
    ax.yaxis.set_major_formatter(PercentFormatter())

    plt.legend()
    plt.title(title)
    plt.tight_layout()

    plt.show()


def plot_all_infected(epicurves, outputdir=None, key="Infected"):
    fig, ax = plt.subplots()

    x = list(sorted(epicurves[list(epicurves.keys())[0]].keys()))

    for name, epicurve in epicurves.items():
        vals, fraction, pct, cumulative = get_all_metrics(epicurve)
        avgs = np.array([np.mean(vals[date][key]) for date in x])
        stdev = np.array([np.std(vals[date][key]) for date in x])
        lines = ax.plot(x, avgs, label=name)
        # ax.fill_between(x, avgs - stdev, avgs + stdev, color=lines[0].get_color())

    plt.xticks(x, map(lambda y: y[5:], x), rotation=90)
    ax.xaxis.set_major_locator(MultipleLocator(7))
    # ax.yaxis.set_major_formatter(PercentFormatter())

    plt.legend()
    plt.title("Total infected per age group")
    plt.tight_layout()

    if outputdir:
        plt.savefig(os.path.join(outputdir, "Infections per age group.png"), dpi=300)

    plt.show()


def get_all_metrics(epicurve):
    vals = defaultdict(lambda: defaultdict(list))
    fraction = defaultdict(lambda: defaultdict(list))
    pct = defaultdict(lambda: defaultdict(list))
    cumulative = defaultdict(lambda: defaultdict(list))

    cumulative_so_far = dict()
    for state in ["Susceptible", "Recovered", "Infected"]:
        cumulative_so_far[state] = dict()
        for i in range(len(epicurve[list(epicurve.keys())[0]]["SUSCEPTIBLE"])):
            cumulative_so_far[state][i] = 0

    for date, values in epicurve.items():
        for i in range(len(values)):
            sus = values["SUSCEPTIBLE"][i]
            inf = values["EXPOSED"][i] + values["INFECTED_SYMPTOMATIC"][i] + values["INFECTED_ASYMPTOMATIC"][i]
            rec = values["RECOVERED"][i]

            total = sus + inf + rec

            cumulative_so_far["Susceptible"][i] += sus
            cumulative_so_far["Infected"][i] += inf
            cumulative_so_far["Recovered"][i] += rec

            vals[date]["Susceptible"].append(sus)
            vals[date]["Infected"].append(inf)
            vals[date]["Recovered"].append(rec)

            fraction[date]["Susceptible"].append(sus / total)
            fraction[date]["Recovered"].append(rec / total)
            fraction[date]["Infected"].append(inf / total)

            pct[date]["Susceptible"].append(sus / total * 100)
            pct[date]["Recovered"].append(rec / total * 100)
            pct[date]["Infected"].append(inf / total * 100)

            cumulative[date]["Susceptible"].append(cumulative_so_far["Susceptible"][i])
            cumulative[date]["Infected"].append(cumulative_so_far["Infected"][i])
            cumulative[date]["Recovered"].append(cumulative_so_far["Recovered"][i])

    return vals, fraction, pct, cumulative


def read_age_stratified_epicurve(simulation_output_directory):
    expected_states = ["NOT_SET", "SUSCEPTIBLE", "EXPOSED", "INFECTED_SYMPTOMATIC", "INFECTED_ASYMPTOMATIC",
                       "RECOVERED"]
    expected_age_groups = ["UNDER_9", "TEEN", "TWENTIES", "THIRTIES", "FORTIES", "FIFTIES", "SIXTIES", "SEVENTIES_PLUS"]

    epicurve = dict()
    age_stratified_epicurves = defaultdict(lambda: defaultdict(dict))
    with open(os.path.join(simulation_output_directory, 'epicurve.sim2apl.csv'), 'r') as epi_in:
        headers = epi_in.readline()[:-1].split(";")
        for line in epi_in:
            line = line[:-1].split(";")
            date = line[0]
            epicurve[date] = dict()
            for state in expected_states:
                epicurve[date][state] = int(line[headers.index(state)])
                for group in expected_age_groups:
                    age_stratified_epicurves[group][date][state] = int(line[headers.index(f"{group}_{state}")])

    return epicurve, age_stratified_epicurves


def age_stratified_epicurve_unit_test(epicurve, age_stratified_epicurves):
    for date, state_values in epicurve.items():
        for state, val in state_values.items():
            aggregate = sum(age_stratified_epicurves[group][date][state] for group in age_stratified_epicurves)
            assert val == aggregate


if __name__ == "__main__":
    plot()
