import datetime
import json
import os.path
import re
import subprocess
import time
from pathlib import Path
from typing import List, Dict

import toml
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.util import UtilityFunction

from classes.ExecutiveOrderOptimizer import NormService
from classes.ExecutiveOrderOptimizer.EOEvaluator import EOEvaluator
from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule
from classes.execution.CodeExecution import CodeExecution
from utility.utility import get_project_root
from .NormService import date_from_data_point


class EOOptimization(CodeExecution):
    rundirectory_template = [
        "optimization",
        "EO0_{x[EO0_start]}_{x[EO0_duration]}-"
        "EO1_{x[EO1_start]}_{x[EO1_duration]}-EO2_{x[EO2_start]}_{x[EO2_duration]}-"
        "EO3_{x[EO3_start]}_{x[EO3_duration]}-EO4_{x[EO4_start]}_{x[EO4_duration]}-"
        "EO5_{x[EO5_start]}_{x[EO5_duration]}-EO6_{x[EO6_start]}_{x[EO6_duration]}-"
        "EO7_{x[EO7_start]}_{x[EO7_duration]}-EO8_{x[EO8_start]}_{x[EO8_duration]}-run{run}",
    ]
    progress_format = "[OPTIMIZATION] [{time}] {ncounties} counties ({fips}): {score} for x={x} policy optimization (dir={output_dir})\n"
    csv_log = os.path.join(get_project_root(), "output", "optimization.results.csv")

    def __init__(
            self,
            societal_global_impact_weight: float,
            norm_weights: str,
            alpha=0.5,
            init_points: int = 20,
            n_iter: int = 160,
            mode_liberal=0.5,
            mode_conservative=0.5,
            fatigue=0.0125,
            fatigue_start=60,
            log_location=None,
            n_slaves=0,
            slave_number=None,
            *args,
            **kwargs
    ):
        super(EOOptimization, self).__init__(*args, **kwargs)
        self.mode_liberal, self.mode_conservative = mode_liberal, mode_conservative
        self.fatigue, self.fatigue_start = fatigue, fatigue_start
        self.run_configuration["liberal"] = self.mode_liberal
        self.run_configuration["conservative"] = self.mode_conservative
        self.run_configuration["fatigue"] = self.fatigue
        self.run_configuration["fatigue_start"] = self.fatigue_start

        self.alpha = alpha
        self.init_points = init_points
        self.n_iter = n_iter
        self.societal_global_impact_weight = societal_global_impact_weight
        self.norm_weights_file = norm_weights
        norm_weights_file_name = os.path.basename(norm_weights)
        self.norm_weights_file_name = norm_weights_file_name[:norm_weights_file_name.rindex(".")]
        self.norm_weights = self.load_norm_weights(self.norm_weights_file)
        self.norm_counts = self.load_norm_application_counts()
        self.county_configuration_file_base = self.county_configuration_file
        self.rundirectory_template.insert(2, self.name)

        self.evaluator = EOEvaluator(societal_global_impact_weight, self.norm_weights, self.norm_counts)

        self.json_log = os.path.join(
            get_project_root(),
            "output",
            f"optimization-alpha{self.alpha}-{self.norm_weights_file_name}-weight{self.societal_global_impact_weight}.json"
        )
        if log_location is not None:
            self.json_log = log_location

        self.n_slaves = n_slaves
        self.slave_number = slave_number

        if self.is_master and (self.n_slaves + 1) % self.n_runs != 0:
            print((self.n_slaves + 1) % self.n_runs)
            raise(Exception(f"The specified number of {self.n_runs} cannot cleanly be distributed across the "
                            f"{self.n_slaves} + 1 master process. Pick another number or create a pull request "
                            f"to deal with this case :')"))

        self.n_simultaneous_runs = int((self.n_slaves + 1) / self.n_runs)

        if self.is_master:
            print(f"Starting as master. Expecting {self.n_slaves} additional slaves")
            self.start_optimization()
        else:
            print(f"Starting as slave {self.slave_number}")
            self.iterate_as_slave()

    def simple_test_f(self, **x):
        """
        Just a simple method that converts the 20 values x can take to a deterministic value using a polynomial function
        """
        val = 0
        x = self.normalize_params(x)
        keys = list(x.keys())
        for i in range(len(keys)):
            val += (i+1) * int(round(x[keys[i]]))
        return -1 * val

    @staticmethod
    def normalize_params(x):
        new_x = dict()
        for key, val in x.items():
            new_x[key] = int(round(val))
        return new_x

    def calibrate(self, **x):
        print(x)
        return super(EOOptimization, self).calibrate(x)

    def start_optimization(self):
        optimizer = BayesianOptimization(
            f=None,  #self.calibrate,  # self.simple_test_f,  # flip around for quick test
            pbounds=NormService.get_bounds(),
            random_state=42,
            verbose=1
        )

        # if os.path.exists(self.json_log):
        #     bayes_opt.util.load_logs(optimizer, [self.json_log])
        # else:
        #     with open(self.json_log, 'w') as outf:
        #         outf.write("")

        # From documentation:
        #   "By default the previous data in the json file is removed. If you want to keep working with the same logger,
        #   the reset paremeter in JSONLogger should be set to False."
        # However, init does not take that parameter (yet?), so we do a workaround
        logger = JSONLogger(path=os.path.join(get_project_root(), ".persistent", ".tmp", "tmp_bayesian_log.json"))
        logger._path = self.json_log

        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.subscribe(Events.OPTIMIZATION_END, logger)

        _logger = ScreenLogger(2)
        optimizer.subscribe(Events.OPTIMIZATION_START, _logger)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, _logger)
        optimizer.subscribe(Events.OPTIMIZATION_END, _logger)

        # "When dealing with functions with discrete parameters,or particularly erratic target space it
        #  might be beneficial to increase the value of the alpha parameter.
        #  This parameters controls how much noise the GP can handle,
        #  so increase it whenever you think that extra flexibility is needed."
        # optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter, alpha=self.alpha, n_restarts_optimizer=0)
        self.maximize(
            optimizer=optimizer,
            init_points=self.init_points,
            n_iter=self.n_iter,
            alpha=self.alpha,
            n_restarts_optimizer=0
        )

        print(optimizer.max)
        print("\nScore", optimizer.max["target"] * -1)
        for key, val in self.normalize_params(optimizer.max['params']).items():
            print("\t", key, val)

    data_points: Dict[str, float] = dict()

    def maximize(self,
                 optimizer: BayesianOptimization,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        optimizer._prime_subscriptions()
        optimizer.dispatch(Events.OPTIMIZATION_START)
        optimizer._prime_queue(init_points)
        optimizer.set_gp_params(**gp_params)

        util = UtilityFunction(
            kind=acq,
            kappa=kappa,
            xi=xi,
            kappa_decay=kappa_decay,
            kappa_decay_delay=kappa_decay_delay
        )
        iteration = 0
        while not optimizer._queue.empty or iteration < n_iter:
            to_probe, number_probed = self.probe_n(optimizer, util, self.n_simultaneous_runs)
            slave = 0
            for i in range(self.n_simultaneous_runs):
                for j in range(self.n_runs):
                    if slave < self.n_slaves:
                        instructions = dict(to_probe[i])
                        instructions["run"] = j
                        self.leave_instructions(slave, instructions)
                    slave += 1

            instructions = dict(to_probe[-1])
            instructions["run"] = self.n_runs - 1
            self.do_optimization_simulation(instructions)

            while not self.all_runs_finished(optimizer):
                print("Waiting for other runs to finish")
                time.sleep(30)

            iteration += number_probed

        optimizer.dispatch(Events.OPTIMIZATION_END)

    def probe_n(self, optimizer, util: UtilityFunction, n):
        x_probes = list()
        probed = 0
        for _ in range(n):
            try:
                x_probe = dict(zip(optimizer._space._keys, next(optimizer._queue)))
            except StopIteration:
                util.update_params()
                x_probe = optimizer.suggest(util)
                probed += 1

            params = self.normalize_params(x_probe)
            self.run_configuration["run"] = 0  # Used in formatting filename that is not used at this point
            self.store_fitness_guess(params)
            if self.run_configuration['policy_schedule_name'] in self.data_points:
                optimizer.register(x_probe, self.data_points[self.run_configuration['policy_schedule_name']])
                print("Params already tested", params)
                probed -= 1
            else:
                x_probes.append(x_probe)

            if optimizer._bounds_transformer:
                optimizer.set_bounds(
                    optimizer._bounds_transformer.transform(optimizer._space))

        return x_probes, probed

    def all_runs_finished(self, optimizer: BayesianOptimization):
        instruction_dir = os.path.join(get_project_root(), ".persistent", ".tmp", self.name)
        for i in range(self.n_slaves):
            if not os.path.exists(os.path.join(instruction_dir, f"run-{i}.done")):
                return False

        for i in range(0, self.n_slaves + 1, self.n_runs):
            with open(os.path.join(instruction_dir, f"run-{i}.done")) as f_in:
                self.deal_with_run(optimizer, json.loads(f_in.read()))

        for i in range(self.n_slaves):
            os.remove(os.path.join(instruction_dir, f"run-{i}.done"))

        return True

    def deal_with_run(self, optimizer: BayesianOptimization, x_probe: Dict[str, float]):
        x_probe_copy = x_probe.copy()
        x_probe_copy.pop('run')
        params = self.normalize_params(x_probe_copy)
        run_directories = dict()
        for i in range(self.n_simultaneous_runs):
            run_directories[i] = os.path.join(*list(map(lambda x: x.format(run=i, x=params), self.rundirectory_template)))
        target, infected, fitness = self.evaluator.fitness([run_directories], NormSchedule(params, "2020-06-28"))
        self.data_points[self.run_configuration['policy_schedule_name']] = target
        optimizer.register(x_probe_copy, target)

    def leave_instructions(self, slave: int, x_probe: Dict[str, int]):
        instruction_dir = os.path.join(get_project_root(), ".persistent", ".tmp", self.name)
        os.makedirs(instruction_dir, exist_ok=True)
        with open(os.path.join(instruction_dir, f"run-{slave}"), 'w') as instructions_out:
            instructions_out.write(json.dumps(x_probe))
            print(f"Created instructions for slave {slave}:", x_probe)

    def do_optimization_simulation(self, x: Dict[str, float]):
        self.run_configuration["run"] = x.pop('run')
        self.store_fitness_guess(x)
        self.start_run_time = datetime.datetime.now()
        self.prepare_simulation_run(x)
        if not os.path.exists(self.get_target_file()):
            print(
                "Starting run ",
                self.run_configuration["run"],
                f"Progress recorded in\n{self._get_agent_run_log_file()}",
            )
            java_command_file = self._create_java_command_file()
            self.set_pansim_parameters(java_command_file)
            self.start_run(java_command_file)
            os.remove(java_command_file)
        else:
            print(
                "\nRun",
                self.run_configuration["run"],
                "already took place; skipping. Delete the directory for that run if it needs to be calculated again",
                self.get_target_file(),
            )

    def iterate_as_slave(self):
        instruction_dir = os.path.join(get_project_root(), ".persistent", ".tmp", self.name)
        instruction_file = os.path.join(instruction_dir, f"run-{self.slave_number}")
        progress_file = os.path.join(instruction_dir, f"run-{self.slave_number}.progress")
        done_file = os.path.join(instruction_dir, f"run-{self.slave_number}.done")

        while(True):
            while not os.path.exists(instruction_file):
                print(f"Slave {self.slave_number} is waiting for instructions")
                print(f"Waiting for file {instruction_file} to appear")
                time.sleep(30)

            xprobe: Dict[str, int]
            with open(instruction_file, 'r') as instructions_in:
                x_probe = json.loads(instructions_in.read())
                print("Instructions found!", x_probe)
            os.rename(instruction_file, progress_file)
            self.do_optimization_simulation(x_probe)
            os.rename(progress_file, done_file)

    @staticmethod
    def load_norm_weights(norm_weights_file) -> Dict[str, float]:
        norm_weights = dict()
        with open(norm_weights_file, 'r') as norm_weights_in:
            norm_weights_in.readline()[:-1].split(",")  # Skip header
            for line in norm_weights_in:
                group = re.findall(r'([\w \[\],;>%]+),([\d.]+)', line)
                if group:
                    norm_weights[group[0][0]] = float(group[0][1])
        return norm_weights

    def store_fitness_guess(self, x):
        x = self.normalize_params(x)
        self.run_configuration['x'] = x
        self.run_configuration['norm_schedule'] = NormSchedule(x, "2020-06-28")
        self.run_configuration['policy_schedule_name'] = os.path.join(
            get_project_root(),
            '.persistent',
            'policies',
            f'{self.alpha}-{self.societal_global_impact_weight}-{self.norm_weights_file_name}',
            "norm-schedule-policy-EO0_{EO0_start}_{EO0_duration}-"
            "EO1_{EO1_start}_{EO1_duration}-EO2_{EO2_start}_{EO2_duration}-"
            "EO3_{EO3_start}_{EO3_duration}-EO4_{EO4_start}_{EO4_duration}-"
            "EO5_{EO5_start}_{EO5_duration}-EO6_{EO6_start}_{EO6_duration}-"
            "EO7_{EO7_start}_{EO7_duration}-EO8_{EO8_start}_{EO8_duration}-run{run}.csv".format(
                run=self.run_configuration['run'], **x
            )
        )

    def prepare_simulation_run(self, x):
        x = self.normalize_params(x)
        self.run_configuration['norm_schedule'].write_to_file(self.run_configuration['policy_schedule_name'])
        toml_config = toml.load(self.county_configuration_file_base)
        toml_config['simulation']['norms'] = self.run_configuration['policy_schedule_name']

        self.county_configuration_file = os.path.join(
            get_project_root(),
            ".persistent",
            'policies',
            f'{self.alpha}-{self.societal_global_impact_weight}-{self.norm_weights_file_name}',
            'configuration',
            "norm-schedule-policy-EO0_{EO0_start}_{EO0_duration}-"
            "EO1_{EO1_start}_{EO1_duration}-EO2_{EO2_start}_{EO2_duration}-"
            "EO3_{EO3_start}_{EO3_duration}-EO4_{EO4_start}_{EO4_duration}-"
            "EO5_{EO5_start}_{EO5_duration}-EO6_{EO6_start}_{EO6_duration}-"
            "EO7_{EO7_start}_{EO7_duration}-EO8_{EO8_start}_{EO8_duration}-run{run}.toml".format(
                run=self.run_configuration["run"], **x
            )
        )

        os.makedirs(Path(self.county_configuration_file).parent.absolute(), exist_ok=True)

        with open(self.county_configuration_file, 'w') as new_conf_out:
            toml.dump(toml_config, new_conf_out)

    def score_simulation_run(self, x: Dict[str, float], directories: List[Dict[int, str]]) -> float:
        target, infected, fitness = self.evaluator.fitness(directories, self.run_configuration['norm_schedule'])
        return target

    def _write_csv_log(self, score):
        pass

    @staticmethod
    def print_data_point_ranges() -> None:
        """
        Quickly print the dates corresponding to the allowed range of data points
        """
        date = datetime.datetime(year=2020, month=3, day=1)
        data_point = 0
        while date < datetime.datetime(year=2020, month=7, day=1):
            data_point += 1
            date = datetime.datetime.strptime(date_from_data_point(data_point), "%Y-%m-%d")

    def load_norm_application_counts(self) -> Dict[str, Dict[str, int]]:
        filename = os.path.abspath(
            os.path.join(
                get_project_root(),
                '.persistent',
                "affected-agents-per-norm-{0}.csv".format("-".join(map(str, sorted(map(lambda x: int(self.counties[x]['fipscode']), self.counties)))))
            )
        )

        if not os.path.exists(filename):
            print("Using behavior model to find what number of agents is affected by what norms")
            self.get_extra_java_commands = lambda: ['--count-affected-agents']
            base_dir_func_backup = self.get_base_directory
            self.get_base_directory = lambda: os.path.abspath(os.path.join(get_project_root(), '.persistent'))
            java_command = self._java_command()
            java_command = java_command[:java_command.index('2>&1')]
            subprocess.run(java_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.get_extra_java_commands = lambda: []
            self.get_base_directory = base_dir_func_backup

        norm_counts = dict()
        with open(filename, 'r') as file_in:
            file_in.readline()  # Skip header
            for line in file_in:
                match = re.findall(r'(\w+(?:\[[\w,;>%]+])?);(\d+);(\d+)', line)
                if match:
                    norm_counts[match[0][0]] = {'affected_agents': int(match[0][1]), 'affected_duration': int(match[0][2])}

        return norm_counts


def test_data_point_range():
    EOOptimization.print_data_point_ranges()
    print("Done")


if __name__ == "__main__":
    # test_fitness()
    for x in range(9):
        print(f"EO{x}_{{x[EO{x}_start]}}_{{x[EO{x}_duration]}}-", end="")
