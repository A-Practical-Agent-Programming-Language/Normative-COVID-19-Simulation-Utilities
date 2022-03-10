import json
import sys
from collections import defaultdict
from typing import Dict, List

from termcolor import colored

from classes.ExecutiveOrderOptimizer.EOOptimization import EOOptimization


class ProgressLog(object):

    def __init__(self, progress_log_file):
        self.progress_log_file = progress_log_file
        self.results, self.by_policy = self.load_progress_log()

    @staticmethod
    def hashable_policy(parameters: Dict[str, float]):
        parameters = EOOptimization.normalize_params(parameters)
        keys = sorted(list(parameters.keys()))
        return tuple([parameters[x] for x in keys])

    def load_progress_log(self):
        results = list()
        by_policy = dict()

        with open(self.progress_log_file, 'r') as progress_in:
            for index, line in enumerate(progress_in):
                result = json.loads(line)
                result["index"] = index
                results.append(result)
                by_policy[self.hashable_policy(result['params'])] = result

        return results, by_policy

    def print_run_scores(self):
        min_score = sys.maxsize
        max_score = min_score * -1

        for result in self.results:
            params = EOOptimization.normalize_params(result['params'])
            l = len(self.by_policy[self.hashable_policy(params)])
            target = float(result['target'])
            score = f"{result['index']}: {-1 * target}. Policy tested {l} times"
            if target > max_score:
                max_score = target
                print(colored(score, 'green'))
            elif target < min_score:
                print(colored(score, 'red'))
                min_score = target
            else:
                print(score)

            if l > 1:
                print(colored(f"\t Policy was tested {len(self.by_policy[self.hashable_policy(params)])} times!"))

            if result['index'] == 149:
                print(colored("================================", 'green'))

    @staticmethod
    def compare_logs(all_logs: List):
        policies = defaultdict(list)
        for log in all_logs:
            for policy in log.results:
                policy["logfile"] = log.progress_log_file
                policies[ProgressLog.hashable_policy(policy['params'])].append(policy)

        for policy, from_logs in policies.items():
            print(policy)
            for log in from_logs:
                print(f"\t {log['logfile']}: {log['target']}")

    def compare_with(self, other_logs: List):
        for result in self.results:
            print(result['index'], result['target'])
            for other_log in other_logs:
                policy = self.hashable_policy(result['params'])
                if policy in other_log.by_policy:
                    other_policy = other_log.by_policy[policy]
                    if other_policy['index'] == result['index']:
                        print(colored(f"\t{other_policy['target']} for {other_log.progress_log_file}", "green"))
                    else:
                        print(colored(f"\t{other_policy['target']} was achieved at index {other_policy['index']} for {other_log.progress_log_file}", "orange"))
                else:
                    print(colored("\t" + other_log.progress_log_file, "red"))


if __name__ == "__main__":
    progress_logs = [ProgressLog(arg) for arg in sys.argv[1:]]
    progress_logs[0].compare_with(progress_logs[1:])
    # for log in progress_logs:
    #     print(log.progress_log_file)
    #     log.print_run_scores()
    # ProgressLog.compare_logs(progress_logs)
