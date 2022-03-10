import json
import os
import re
from collections import defaultdict
from typing import Dict

import numpy as np

from classes.ExecutiveOrderOptimizer.EOEvaluator import EOEvaluator
from classes.ExecutiveOrderOptimizer.EOOptimization import EOOptimization
from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule
from classes.execution.NormExperiment import NormExperiment
from utility.utility import get_project_root
from scripts.plot_epicurve_average import EpicurvePlotter

calculate_new_omega_c = False
# omega_c = 0.001273665483847564   # Previously: 100/7
omega_c = 0.0022626371521808136
affected_agents_file = 'affected-agents-per-norm-65-75-109-540.csv'
actually_implemented_policy = "/home/jan/dev/university/2apl/simulation/sim2apl-episimpledemics/src/main/resources/norm-schedule.csv"
# experiment_results = "/home/jan/dev/university/2apl/simulation/calibration/output/remote-agent-runs/scratch-backup-after-mabs/journal-experiment-master-slave-runs-10-including-louisa/experiment/4counties-fips-109-540-65-75"
# experiment_results = "/home/jan/dev/university/2apl/simulation/calibration/output/remote-agent-runs/bayesian-optimization/after-holiday-fix/experiment"
experiment_results = "/home/jan/dev/university/2apl/simulation/calibration/output/remote-agent-runs/optimization/experiment-120-steps/4counties-fips-109-540-65-75"
# optimization_results = "/home/jan/dev/university/2apl/simulation/calibration/output/remote-agent-runs/bayesian-optimization/after-holiday-fix/optimization"
optimization_results = "/home/jan/dev/university/2apl/simulation/calibration/output/remote-agent-runs/optimization/optimization"

max_simulation_date = "2020-06-28"

##############################################################
#
#              Load the static data
#
##############################################################
norm_weights = {
    "default-weights": os.path.join(get_project_root(), 'external', 'norm_weights.csv'),
    "favour-schools": os.path.join(get_project_root(), 'external', 'norm_weights_favour_schools_open.csv'),
    "favour-economy": os.path.join(get_project_root(), 'external', 'norm_weights_favour_economy.csv')
}
weight_to_letter_map = {"default-weights": "D", "favour-economy": "E", "favour-schools": "S"}


def science_exp(number, precision: int = 2):
    number, exponent = f"{number:.{precision}e}".split("e+")
    number = float(number)
    exponent = int(exponent)
    return f"${number}\cdot 10^{{{exponent}}}$"


def load_evaluator(global_weight: float, norm_weights_file: str, affected_agents_file_arg):
    norm_counts = dict()
    with open(os.path.join(get_project_root(), '.persistent', affected_agents_file_arg),
              'r') as file_in:
        file_in.readline()  # Skip header
        for line in file_in:
            match = re.findall(r'(\w+(?:\[[\w,;>%]+])?);(\d+);(\d+)', line)
            if match:
                norm_counts[match[0][0]] = {'affected_agents': int(match[0][1]),
                                            'affected_duration': int(match[0][2])}

    return EOEvaluator(
        societal_global_impact_weight=global_weight,
        norm_weights=EOOptimization.load_norm_weights(norm_weights_file),
        norm_counts=norm_counts
    )


evaluators = dict()
for weight_name, weight_file in norm_weights.items():
    evaluators[weight_name] = load_evaluator(omega_c, norm_weights[weight_name], affected_agents_file)

possible_weights = list()


def read_experiment_directories(experiments_directory):
    dates = NormExperiment.get_experiment_dates(actually_implemented_policy, max_simulation_date)
    experiment_targets = defaultdict(lambda: defaultdict(dict))

    for norm_weight_group, evaluator in evaluators.items():
        # for subdir in sorted(os.listdir(experiments_directory)):
        #     directory = os.path.join(experiments_directory, subdir)
        #     if os.path.isdir(directory):
        #         m = re.findall(r'experiment-(\d+)-norms-until(\d{4}-\d{2}-\d{2})-run(\d+)', subdir)
        #         if len(m) and m[0][1] < max_simulation_date:
        #             last_eo_date = dates[int(m[0][0])]
        #             norm_schedule = NormSchedule.from_norm_schedule(
        #                 actually_implemented_policy,
        #                 last_simulation_date="2020-06-28",
        #                 until=last_eo_date
        #             )
        #
        #             target, infected, fitness = evaluator.fitness([{0: directory}], norm_schedule)
        #             epicurve = EpicurvePlotter.read_epicurve_file(directory)[0][max_simulation_date]
        #             population_size = sum([epicurve[x] for x in epicurve if type(epicurve[x]) == int])
        #             experiment_targets[m[0][0]][norm_weight_group]['target'].append(-1 * target)
        #             experiment_targets[m[0][0]][norm_weight_group]['infections_total'].append(infected)
        #             experiment_targets[m[0][0]][norm_weight_group]['infections_pct'].append(infected / population_size)
        #
        #             if norm_weight_group == "default-weights" and "experiment-9" in subdir:
        #                 new_omega_c = infected / fitness
        #                 print(f"{subdir[-4:]}: 𝜔𝑐 = {new_omega_c}, ==> infected = 𝜔𝑐 * fitness ==== {infected} = {new_omega_c} * {fitness} ==== {infected} = {new_omega_c * fitness}")
        #                 possible_weights.append(new_omega_c)
        grouped_directories = group_experiment_directories(experiments_directory)
        for experiment, directories in grouped_directories.items():
            last_eo_date = dates[experiment]
            norm_schedule = NormSchedule.from_norm_schedule(
                actually_implemented_policy, last_simulation_date="2020-06-28", until=last_eo_date
            )
            target, infected, fitness = evaluator.fitness([directories], norm_schedule)
            epicurve = EpicurvePlotter.read_epicurve_file(directories[0])[0][max_simulation_date]
            population_size = sum([epicurve[x] for x in epicurve if type(epicurve[x]) == int])
            experiment_targets[experiment][norm_weight_group] = dict(
                target=-1 * target,
                infections_total=infected,
                infections_pct=(infected / population_size)
            )

            if calculate_new_omega_c and  norm_weight_group == "default-weights" and experiment == max(grouped_directories.keys()):
                new_omega_c = infected / fitness
                print(f"𝜔𝑐 = {new_omega_c}, ==> infected = 𝜔𝑐 * fitness ==== {infected} = {new_omega_c} * {fitness} ==== {infected} = {new_omega_c * fitness}")
                possible_weights.append(new_omega_c)

    return experiment_targets


def group_experiment_directories(experiments_directory):
    experiment_directories = defaultdict(dict)
    for subdir in os.listdir(experiments_directory):
        m = re.findall(r'experiment-(\d+)-norms-until(\d{4}-\d{2}-\d{2})-run(\d+)', subdir)
        if len(m) and m[0][1] < max_simulation_date:
            experiment_directories[int(m[0][0])][int(m[0][2])] = os.path.join(experiments_directory, subdir)

    return experiment_directories

##############################################################
#
#               READ OPTIMIZED DATA
#
##############################################################


def group_optimization_runs_by_weight(
        optimization_directory: str
) -> Dict[str, Dict[float, Dict[str, str or Dict[int, str]]]]:
    runs_by_weight = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for directory in os.listdir(optimization_directory):
        abs_dir = os.path.join(optimization_directory, directory)
        if os.path.isdir(abs_dir):
            for subdir in os.listdir(abs_dir):
                abs_subdir = os.path.join(optimization_directory, directory, subdir)
                if os.path.isdir(abs_subdir) and subdir.startswith("alpha-"):
                    alpha = float(subdir[len("alpha-"):])
                    for run_dir in os.listdir(abs_subdir):
                        match = re.findall(r'run(\d+)', run_dir)
                        if match:
                            run = int(match[0])
                            key = 'runs' if os.path.isdir(os.path.join(abs_subdir, run_dir)) else 'schedules'
                            runs_by_weight[directory][alpha][key][run] = os.path.join(abs_subdir, run_dir)
                        elif run_dir.endswith(".json") and "optimization" in run_dir:
                            runs_by_weight[directory][alpha]['progresslog'] = os.path.join(abs_subdir, run_dir)

    return runs_by_weight


def load_optimization_results(
        runs_by_weight: Dict[str, Dict[float, Dict[str, str or Dict[int, str]]]]
) -> Dict[str, Dict[float, Dict[str, int or float or str]]]:
    optimization_outcomes_by_weight = defaultdict(lambda: defaultdict(dict))

    for weight, weight_dict in runs_by_weight.items():
        for alpha, information in weight_dict.items():
            optimization_outcomes_by_weight[weight][alpha]['data'] = list()
            with open(information['progresslog'], 'r') as json_in:
                for line in json_in:
                    data = json.loads(line)
                    data['params'] = EOOptimization.normalize_params(data['params'])
                    optimization_outcomes_by_weight[weight][alpha]["data"].append(data)

            optimization_outcomes_by_weight[weight][alpha]['norm-schedule'] = information['schedules'][0]

            # something = EpicurvePlotter.read_epicurve_file(information['runs'][0])
            # epicurve = something[0][max_simulation_date]
            _, population_size = EpicurvePlotter.read_epicurve_file(information['runs'][0])  # sum([epicurve[x] for x in epicurve if type(epicurve[x]) == int])
            infected = EOEvaluator.count_infected_agents([information['runs']])
            optimization_outcomes_by_weight[weight][alpha]['infections_total'] = infected
            optimization_outcomes_by_weight[weight][alpha]['infections_pct'] = infected / population_size

    return optimization_outcomes_by_weight


def read_optimization_directories(optimization_directory) -> Dict[float, Dict[str, any]]:
    runs_by_weight = group_optimization_runs_by_weight(optimization_directory)
    optimization_results = load_optimization_results(runs_by_weight)

    optimization_outcomes = defaultdict(lambda: dict())

    for weight, weight_dict in optimization_results.items():
        for alpha, data_list in weight_dict.items():
            highest_target = max(data_list["data"], key=lambda x: x["target"])
            schedule = NormSchedule(highest_target['params'], '2020-06-28')

            target, _, _ = evaluators[weight].fitness([runs_by_weight[weight][alpha]['runs']], schedule)

            optimization_outcomes[weight]['pct'] = data_list['infections_pct'] * 100
            optimization_outcomes[weight]['target'] = -1*highest_target["target"]
            optimization_outcomes[weight]['target_recalculated'] = target

    return optimization_outcomes


def print_table_vertically(
        experiment_outcomes,
        optimization_outcomes,
        included_weights=['default-weights', 'favour-economy', 'favour-schools']
    ):
    print("$P_\\star$", end='')

    for weight in included_weights:
        if weight in optimization_outcomes:
            print(f"& {science_exp(optimization_outcomes[weight]['target'])} & ${optimization_outcomes[weight]['pct']:.1f}\%$", end='')
        else:
            print("& &", end='')
    print(" \\\\ \midrule")

    print("$P_{\mathit{Virginia}}$", end='')
    for weight in included_weights:
        actual_policy = max(experiment_outcomes.keys())
        target = np.mean(experiment_outcomes[actual_policy][weight]['target'])
        infection_pct = np.mean(experiment_outcomes[actual_policy][weight]['infections_pct'])
        print(f" & {science_exp(target)} & ${infection_pct*100:.1f}\%$", end='')
    print(" \\\\ \midrule")

    for exp in sorted(experiment_outcomes):
        print(f"P$_{{{exp}}}$", end='')
        for weight in included_weights:
            target = np.mean(experiment_outcomes[exp][weight]['target'])
            infection_pct = np.mean(experiment_outcomes[exp][weight]['infections_pct'])
            print(f" & {science_exp(target)} & ${infection_pct*100:.1f}\%$", end='')
        print(" \\\\")
    print("\\bottomrule")


def print_table_horizontally(
        experiment_outcomes,
        optimization_outcomes,
        included_weights=['default-weights', 'favour-economy', 'favour-schools']
    ):

    # Headers & start
    # TODO, if more than one weight, add extra column to start
    print("\\resizebox{\linewidth}{!}{ %")
    print("\\begin{tabular}{", end='')
    if len(included_weights) > 1:
        print("l", end='')
    print("r|" + "l"*(len(experiment_outcomes)+2) + "}")
    if len(included_weights) > 1:
        print(" & ", end='')
    print("& $P_\star$ & $P_{\mathit{Virginia}}$", end='')
    for exp in experiment_outcomes:
        print(f" & $P_{{{exp}}}$", end='')
    print("\\\\ \\midrule")

    actual_policy = max(experiment_outcomes.keys())

    for i, weight in enumerate(included_weights):
        if len(included_weights) > 1:
            print(f"\\multirow{{2}}{{*}}{{${weight_to_letter_map[weight]}$}} &", end='')

        # Print target scores
        print("$f(P)$ & ", end='')
        if weight in optimization_outcomes:
            print(f"{science_exp(optimization_outcomes[weight]['target'])}", end='')
        print(f" & {science_exp(np.mean(experiment_outcomes[actual_policy][weight]['target']))}", end='')
        for exp in experiment_outcomes:
            print(f" & {science_exp(np.mean(experiment_outcomes[exp][weight]['target']))}", end='')
        print("\\\\")

        # Print infection rates
        if len(included_weights) > 1:
            print("& ", end='')
        print("inf ($0$) & ", end='')
        if weight in optimization_outcomes:
            print(f"${optimization_outcomes[weight]['pct']: .1f}\% $", end='')
        print(f" & ${np.mean(experiment_outcomes[actual_policy][weight]['infections_pct']) * 100:.1f}\%$", end='')
        for exp in experiment_outcomes:
            print(f" & ${np.mean(experiment_outcomes[exp][weight]['infections_pct']) * 100:.1f}\%$", end='')
        print("\\\\", end='')
        print("\\bottomrule" if i == len(included_weights) - 1 else "\\midrule")

    print("\\end{tabular}")
    print("}")


# def verify_score_calculations():
#     for weight, weight_dict in optimization_results:
#         print(weight)
#         for alpha, data_list in weight_dict.items():
#             print("\t", alpha)
#             print("\t\tinfected: ", data_list["infections_total"],
#                   "({0:.1f}%)".format(data_list["infections_pct"] * 100))
#             print(
#                 f"\t\tbest: {round(-1 * max(target_list)):.2e}\t(recalculated to verify consistent outcome: {round(fitness_target * -1):.2e}. Difference is {max(target_list) - fitness_target:.3e}")
#             print("\t\t\tEO0_{EO0_start}_{EO0_duration}-"
#                   "EO1_{EO1_start}_{EO1_duration}-EO2_{EO2_start}_{EO2_duration}-"
#                   "EO3_{EO3_start}_{EO3_duration}-EO4_{EO4_start}_{EO4_duration}-"
#                   "EO5_{EO5_start}_{EO5_duration}-EO6_{EO6_start}_{EO6_duration}-"
#                   "EO7_{EO7_start}_{EO7_duration}-EO8_{EO8_start}_{EO8_duration}".format(
#                     **target_dct[max(target_list)]['params']
#                     )
#             )

if __name__ == "__main__":
    experiment_outcomes = read_experiment_directories(experiment_results)
    optimization_outcomes = read_optimization_directories(optimization_results)

    print_table_vertically(experiment_outcomes, optimization_outcomes)
    # print_table_horizontally(experiment_outcomes, optimization_outcomes, ['default-weights'])

    if calculate_new_omega_c:
        print(f"Final suggestion: 𝜔𝑐 = {np.mean(possible_weights)}  +/-  {np.std(possible_weights)}")
        print(f"Previously 𝜔𝑐 =100/7 = {100/7:3f}, so now 𝜔𝑐 is (100/7)/𝜔𝑐 = {(100/7)/np.mean(possible_weights)} times smaller")
