"""
This script serves to extend the progress logs created by the Bayesian Optimization with the target score of each
individual run. By default, the progress logs only contain the average of 5 runs for any policy.

The script requires the full simulation output for each simulation to be present.

The script assumes the following directory hierarchy.
* As the first argument to this script, the working directory is passed.
* Within the working directory, there are the progress log files, with a name starting with "bayesian-optimization" and
    having the file extension ".json".
    Moreover, these files contain a string "norm_weights_{identifier}-progress-log" followed by an identifier that also
    occurs in the following path names:
* Within the working directory, there is a sub directory called "all-runs/optimization", in which all the different
    optimization processes for which a progress file exists are present. Within each of those directories are all the
    simulation runs for that optimization process, with the policy as their name
"""
import json
import os
import re
import sys
from collections import defaultdict

from classes.ExecutiveOrderOptimizer.EOEvaluator import EOEvaluator
from classes.ExecutiveOrderOptimizer.EOOptimization import EOOptimization
from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule
from classes.ExecutiveOrderOptimizer.NormService import norms_to_new_name_map
from utility.utility import get_project_root

wd = sys.argv[1]

progress_logs = [
    os.path.join(wd, x)
    for x in os.listdir(wd)
    if x.startswith("bayesian-optimization")
    and x.endswith(".json")
]

all_runs = list()

for log in progress_logs:
    match = re.findall(r'norm_weights_(\w+)-progress-log', log)
    if not match:
        match = ['default-norm-weights']
    if match:
        norm_weights_file = "norm_weights.csv"
        if "schools" in match[0]:
            norm_weights_file = "norm_weights_favour_schools_open.csv"
        elif "economy" in match[0]:
            norm_weights_file = "norm_weights_favour_economy.csv"
        norm_counts = dict()
        with open(
                os.path.join(get_project_root(), ".persistent", "affected-agents-per-norm-65-75-109-540.csv"),
                'r'
        ) as file_in:
            file_in.readline()  # Skip header
            for line in file_in:
                norm_match = re.findall(r'(\w+(?:\[[\w ,;>%]+])?);(\d+);(\d+)', line)
                if norm_match:
                    norm_counts[norm_match[0][0]] = {
                        'affected_agents': int(norm_match[0][1]),
                        'affected_duration': int(norm_match[0][2])
                    }
        eoEvaluator = EOEvaluator(
            0.0007628406109972039,  # 0.0022626371521808136,
            EOOptimization.load_norm_weights(os.path.join(get_project_root(), "external", norm_weights_file)),
            norm_counts
        )
        for simulation_directory in os.listdir(os.path.join(wd, "all-runs", "optimization")):
            if match[0] in simulation_directory or match[0].replace("_", "-") in simulation_directory:
                all_runs.append((
                    match[0],
                    log,
                    os.path.join(wd, "all-runs", "optimization", simulation_directory),
                    eoEvaluator
                ))
                break

rundirectory_template = [
    "optimization",
    "EO0_{x[EO0_start]}_{x[EO0_duration]}-"
    "EO1_{x[EO1_start]}_{x[EO1_duration]}-EO2_{x[EO2_start]}_{x[EO2_duration]}-"
    "EO3_{x[EO3_start]}_{x[EO3_duration]}-EO4_{x[EO4_start]}_{x[EO4_duration]}-"
    "EO5_{x[EO5_start]}_{x[EO5_duration]}-EO6_{x[EO6_start]}_{x[EO6_duration]}-"
    "EO7_{x[EO7_start]}_{x[EO7_duration]}-EO8_{x[EO8_start]}_{x[EO8_duration]}-run{run}",
]

all_policies = list()

for name, log, directory, evaluator in all_runs:
    policies = defaultdict(dict)
    with open(os.path.join('/home/jan/dev/university/2apl/simulation/bayesian-optimization-cost/progress-logs', f'{name}.json'), 'w') as log_out:
    # with open(os.path.join(wd, 'all-runs', f'{name}-with-extra-progress-log.json'), 'w') as log_out:
        with open(log, 'r') as log_in:
            for line in log_in:
                data = json.loads(line)
                params = EOOptimization.normalize_params(data['params'])
                ns = NormSchedule(params, "2020-06-28")
                extra_data = dict()
                targets = list()
                runs = dict()
                for i in range(5):
                    simulation_run = os.path.join(
                        directory,
                        rundirectory_template[1].format(run=i, x=params)
                    )
                    target, infected, fitness = evaluator.fitness([{0: simulation_run}], ns)
                    targets.append(target)
                    extra_data[f"run-{i}"] = dict(target=target, infected=infected, cost=fitness)
                    runs[i] = simulation_run
                target, infected, fitness = evaluator.fitness([runs], ns)

                policies[ns] = dict(target=target, infected=infected, cost=fitness)

                data["target"] = target,
                data["infected"] = infected,
                data["cost"] = fitness
                data["params"] = ns.get_norm_event_matrix()

                data["per-run-results"] = extra_data

                for norm in data["params"]:
                    from_ns = ns.get_active_duration(norm)
                    from_matrix = sum([7 for x in data["params"][norm] if x])
                    try:
                        assert from_ns == from_matrix, f"Expected {from_ns} (from norm schedule) and {from_matrix} (from matrix) to be equal for norm {norm}"
                    except AssertionError as e:
                        print(e)

                log_out.write(json.dumps(data))
                log_out.write("\n")
                all_policies.append(dict(params=data["params"], infected=data["infected"]))

    best_policies = sorted(list(policies.keys()), key=lambda x: policies[x]['target'], reverse=True)
    best_policy_vals = policies[best_policies[0]]
    print("With weights {name}, the best policy had a target of {target} with {infected} infected and total cost {cost}:".format(name=name, **best_policy_vals))
    best_policies[0].prettyprint(extended=True)
    print(dict(best_policies[0].get_norm_event_matrix()))
    print("\n\n")

print(len(all_policies))

with open(os.path.join(wd, 'all-runs', 'all-policies.json'), 'w') as all_policies_out:
    json.dump(all_policies, all_policies_out)

policy_length = [len(policy['params']) for policy in all_policies]
print(len(policy_length), sorted(set(policy_length)))


def copy_file_removing_unused_norms(fin, fout):
    skipped, copied = 0, 0
    with open(fin, 'r') as fin_stream:
        with open(fout, 'w') as fout_stream:
            fout_stream.write(fin_stream.readline())  # Copy header
            for line in fin_stream:
                if line.strip():
                    norm = re.findall(r'(\w+(?:\[.*])?)', line)[0]
                    if norm in norms_to_new_name_map:
                        copied += 1
                        fout_stream.write(line)
                        if not line.endswith("\n"):
                            fout_stream.write("\n")
                    else:
                        skipped += 1
                        print(f"Skipping norm {norm}")

    print(f"{copied} norms copied, skipped {skipped}")


copy_file_removing_unused_norms(
    get_project_root('.persistent/affected-agents-per-norm-65-75-109-540.csv'),
    '/home/jan/dev/university/2apl/simulation/bayesian-optimization-cost/affected-agents-per-norm-65-75-109-540.csv'
)

for f in os.listdir(get_project_root('external')):
    if f.startswith("norm_weights"):
        copy_file_removing_unused_norms(
            get_project_root('external', f),
            f'/home/jan/dev/university/2apl/simulation/bayesian-optimization-cost/{f}'
        )

with open(os.path.join(wd, 'all-runs', 'all-policies.json'), 'w') as all_policies_out:
    all_policies_out.write("[")
    all_policies_out.write(",\n".join(map(json.dumps, all_policies)))
    all_policies_out.write("]")





