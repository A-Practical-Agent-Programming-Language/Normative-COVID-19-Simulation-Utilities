import json
import os.path
import re
import time
from datetime import datetime
from typing import List, Dict

import numpy as np
import toml

import utility.utility
from classes.ExecutiveOrderOptimizer.EOEvaluator import EOEvaluator
from classes.ExecutiveOrderOptimizer.NormSchedule import NormSchedule
from classes.ExecutiveOrderOptimizer.NormService import DATE_FORMAT
from classes.ExecutiveOrderOptimizer.covid_policy_bayes_opt.bayes_policy_opt import BayesOptMinimizer, FloatArray, \
    split_X, digitize_D, IntArray, \
    digitize_S, WaitingForFinalEvals, MinimizationComplete
from classes.execution.CodeExecution import CodeExecution
from utility.utility import get_project_root


class EOOptimization(CodeExecution):
    rundirectory_template = [
        "optimization",
        "policy-{serialized_x}-run-{run}"
    ]
    progress_format = "[OPTIMIZATION] [{time}] {ncounties} counties ({fips}): {score} for x={x} policy optimization (dir={output_dir})\n"
    csv_log = os.path.join(get_project_root(), "output", "optimization.results.csv")

    def __init__(
            self,
            societal_global_impact_weight: float,
            policy_specification: str,
            init_points: int = 256,
            explore_evals: int = 32,
            exploit_evals: int = 32,
            kappa_initial: float = 2.576,
            kappa_scale: float = 0.95,
            mode_liberal=0.5,
            mode_conservative=0.5,
            fatigue=0.0125,
            fatigue_start=60,
            log_location=None,
            *args,
            **kwargs
    ):
        super(EOOptimization, self).__init__(*args, **kwargs)
        self.target_file = "epicurve.sim2apl.csv"
        self.mode_liberal, self.mode_conservative = mode_liberal, mode_conservative
        self.fatigue, self.fatigue_start = fatigue, fatigue_start
        self.run_configuration["liberal"] = self.mode_liberal
        self.run_configuration["conservative"] = self.mode_conservative
        self.run_configuration["fatigue"] = self.fatigue
        self.run_configuration["fatigue_start"] = self.fatigue_start
        self.instruction_dir = os.path.join(get_project_root(), ".persistent", ".tmp", self.name)

        self.initial_evaluations = init_points
        self.explore_evals = explore_evals
        self.exploit_evals = exploit_evals
        self.kappa_initial = kappa_initial
        self.kappa_scale = kappa_scale
        self.societal_global_impact_weight = societal_global_impact_weight

        self.policy_specification_file = policy_specification
        policy_specification_file_name = os.path.basename(policy_specification)
        self.policy_specification_file_name = policy_specification_file_name[:policy_specification_file_name.rindex(".")]

        with open(policy_specification, 'r') as policy_specification_in:
            self.policy = json.load(policy_specification_in)
        self.county_configuration_file_base = self.county_configuration_file
        self.rundirectory_template.insert(2, self.name)

        self.static_penalty, self.dynamic_penalty = self.get_penalty()
        self.static_policy_nvals = np.array([len(v) for v in self.static_penalty])
        self.n_weeks = self.get_n_weeks()
        self.evaluator = EOEvaluator(societal_global_impact_weight, policy_specification=self.policy)

        simulation_end: datetime = utility.utility.get_expected_end_date(self.county_configuration_file, self.n_steps)
        self.simulation_end = simulation_end.strftime(DATE_FORMAT)

        self.json_log = os.path.join(
            get_project_root(),
            "output",
            f"optimization-{self.policy_specification_file_name}-weight{self.societal_global_impact_weight}.json"
        )
        if log_location is not None:
            self.json_log = log_location

        print("Things should be happening now")
        if self.is_master and (self.n_slaves + 1) % self.n_runs != 0:
            print((self.n_slaves + 1) % self.n_runs)
            raise(Exception(f"The specified number of {self.n_runs} cannot cleanly be distributed across the "
                            f"{self.n_slaves} + 1 master process. Pick another number or create a pull request "
                            f"to deal with this case :')"))
        elif self.is_master:
            print("Starting optimization master process")
            self.calibrate()
        else:
            print(f"Starting optimization slave process {self.slave_number}")
            self.iterate_as_slave()

    def create_static_data_object(self, base_path, now):
        # Creates an object to be exported to JSON at the start of the run, from which used parameters can be reconstructed
        data = super().create_static_data_object(base_path, now)
        data["societal_global_impact_weight"] = self.societal_global_impact_weight
        data["initial_evaluations"] = self.initial_evaluations
        data["explore_evals"] = self.explore_evals,
        data["exploit_evals"] = self.exploit_evals
        data["static_penalty"] = [list(v) for v in self.static_penalty]
        data["dynamic_penalty"] = self.dynamic_penalty
        data["K_(n_weeks)"] = self.n_weeks
        data["kappa_initial"] = self.kappa_initial
        data["kappa_scale"] = self.kappa_scale
        return data

    def get_penalty(self) -> (List[FloatArray], float):
        """
        Returns: The static and dynamic penalties based on the provided policy specification
        """
        static_penalty = [np.array([x['penalty'] for x in self.policy['static'][norm]], dtype=float) for norm in sorted(self.policy['static'])]
        dynamic_penalty = self.policy['dynamic']['StayHomeSick'][1]['penalty']
        return static_penalty, dynamic_penalty

    def get_n_weeks(self) -> int:
        """
        Returns: The number of weeks (I.e., the value K) the simulation will be running
        """
        return int(self.n_steps / 7)

    def get_files_to_persist(self):
        """
        Returns: The files that should be saved at the start of this simulation, so that we can reconstruct configuration later
        """
        files = super().get_files_to_persist()
        files += [
            self.policy_specification_file
        ]
        return files

    def calibrate(self, **x):
        minimizer = BayesOptMinimizer(
            static_penalty=self.static_penalty,
            dynamic_penalty=self.dynamic_penalty,
            M=len(self.static_penalty),
            K=self.n_weeks,
            omega=self.societal_global_impact_weight,
            init_evals=self.initial_evaluations,
            explore_evals=self.explore_evals,
            exploit_evals=self.exploit_evals,
            parallel_evals=self.n_slaves + 1,
            kappa_initial=self.kappa_initial,
            kappa_scale=self.kappa_scale,
        )

        # TODO, if exists, load state dict

        processed = self.load_existing_runs(minimizer)

        self.start_optimization(minimizer, processed)

    def start_optimization(self, minimizer: BayesOptMinimizer, already_processed: int = 0):
        finished = False

        initial_xs: List[FloatArray] = minimizer.get_initial_xs()

        remaining_to_process = len(initial_xs) - already_processed
        print(f"Received {len(initial_xs)} initial xs, but already processed {already_processed}. Performing remaining first {remaining_to_process} from initial xs")
        initial_xs = initial_xs[:remaining_to_process]
        print(len(initial_xs))

        for x_probes in [initial_xs[i:i+self.n_slaves+1] for i in range(0, len(initial_xs), self.n_slaves+1)]:
            self.handle_simultaneous_probes(minimizer, x_probes)

        self.write_lines_to_progress_log([f"Finished initial {len(initial_xs)} simulations"])

        while not finished:
            to_probe = list()
            for _ in range(self.n_slaves + 1):
                try:
                    to_probe.append(minimizer.get_next_x())
                except WaitingForFinalEvals:
                    print("Final simulations will start")
                    finished = True
                except MinimizationComplete:
                    print("These probes should not have happened!")
                    finished = True

            if len(to_probe):
                self.handle_simultaneous_probes(minimizer, to_probe)

        print("Simulation finished!")
        best_x, penalty_mean, penalty_std = minimizer.get_best_pred()
        print("Best policy was", best_x)
        print(f"Penalty mean={penalty_mean} and std={penalty_std}")
        self.write_lines_to_progress_log([
            "Optimization finished",
            "Best predicted policy:",
            f"\tPenalty mean: {penalty_mean}, penalty std: {penalty_std}, {self.serialize_policy(best_x)}, {best_x}"
        ])

        for slave in range(self.n_slaves):
            self.leave_instructions(slave, best_x, run=slave)
        self.do_optimization_simulation(best_x, self.n_slaves)
        while not self.all_runs_finished(minimizer, [best_x] * (self.n_slaves + 1))[0]:
            print("Master is waiting for slaves to finish. Checking again in 30 seconds")
            time.sleep(30)
        self.write_lines_to_progress_log([f"Finished {self.n_slaves + 1} simulations of best predicted X"])

    def handle_simultaneous_probes(self, optimizer: BayesOptMinimizer, x_probes: List[FloatArray]):
        for x_probe, slave in zip(x_probes[1:], range(self.n_slaves)):
            if not self.simulation_exists(x_probes[slave], run=0)[0]:
                print(f"Leaving instructions to simulate {self.serialize_policy(x_probe)}")
                self.leave_instructions(slave, x_probe, run=0)
            else:
                print(f"Not leaving instructions for slave {slave}; The simulation path already exists")

        print(f"Simulating {self.serialize_policy(x_probes[0])} on master process")
        self.do_optimization_simulation(x_probes[0], 0)

        while not self.all_runs_finished(optimizer, x_probes)[0]:
            print("Master is waiting for slaves to finish. Checking again in 30 seconds")
            time.sleep(30)

    def do_optimization_simulation(self, x: FloatArray, run: int):
        self.run_configuration["run"] = run
        self.prepare_simulation_run(x)

        if not os.path.exists(self.get_target_file(0)):
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

    def prepare_simulation_run(self, x: FloatArray):
        self.run_configuration['x'] = x
        serialized_x = self.serialize_policy(x)
        self.run_configuration["serialized_x"] = serialized_x
        run = self.run_configuration["run"]
        base = self.get_base_directory(run)
        os.makedirs(base, exist_ok=True)

        self.run_configuration['policy_schedule_name'] = self.get_base_directory(
            run,
            f"norm-schedule-policy-{serialized_x}-run-{run}.csv"
        )
        self.county_configuration_file = self.get_base_directory(
            run,
            f"run-configuration-policy_{serialized_x}-run-{run}.csv"
        )

        toml_config = toml.load(self.county_configuration_file_base)
        toml_config['simulation']['norms'] = self.run_configuration['policy_schedule_name']

        with open(self.county_configuration_file, 'w') as new_conf_out:
            toml.dump(toml_config, new_conf_out)

        with open(self.get_base_directory(run, "x-probe.txt"), 'w') as xprobe_out:
            xprobe_out.write(f"{x}")

        self.run_configuration['norm_schedule'] = self.x_to_norm_schedule(x)
        self.run_configuration['norm_schedule'].write_to_file(self.run_configuration['policy_schedule_name'])

    def simulation_exists(self, x_probe: FloatArray, run: int) -> (bool, os.PathLike):
        params = self.serialize_policy(x_probe)
        path = os.path.join(*list(map(lambda t: t.format(run=run, serialized_x=params), self.rundirectory_template)))
        exists = os.path.exists(path)
        return exists, path

    def check_instructions_finished(self, x_probe: FloatArray, slave: int) -> bool:
        base, progress, done = [os.path.exists(os.path.join(self.instruction_dir, f"run-{slave}{t}")) for t in ["", ".progress", ".done"]]
        return done or (not base and not progress and self.simulation_exists(x_probe, slave)[0])

    def all_runs_finished(self, optimizer: BayesOptMinimizer, x_probes: List[FloatArray]) -> (bool, List[str]):
        not_finished = list()
        for i in range(self.n_slaves):
            if i <= len(x_probes) and not self.check_instructions_finished(x_probes[i], i):
                not_finished.append(str(i))

        if len(not_finished):
            return False, not_finished

        for x_probe in x_probes:
            self.deal_with_run(optimizer, x_probe)

        for i in range(self.n_slaves):
            if os.path.exists(os.path.join(self.instruction_dir, f"run-{i}.done")):
                os.remove(os.path.join(self.instruction_dir, f"run-{i}.done"))

        return True, []

    def deal_with_run(self, optimizer: BayesOptMinimizer, x_probe: FloatArray):
        params = self.serialize_policy(x_probe)
        run_directories = dict()
        for i in range(self.n_runs):
            run_directories[i] = os.path.join(*list(map(lambda x: x.format(run=i, x=x_probe, serialized_x=params), self.rundirectory_template)))

        target, infected, fitness = self.evaluator.fitness(
            [run_directories],
            self.x_to_norm_schedule(x_probe)
        )

        optimizer.set_iota(x_probe, infected)

        self.write_lines_to_progress_log([f"{target}, {infected}, {fitness}, {self.serialize_policy(x_probe)} {x_probe}"])

    def write_lines_to_progress_log(self, lines: List[str]):
        o = os.path.join(*self.rundirectory_template[:-1], "optimization.progress.log")
        with open(o, 'a+') as progress_out:
            for line in lines:
                progress_out.write(line + "\n")

    def load_existing_runs(self, minimizer: BayesOptMinimizer):
        matcher = re.compile(r'^-?[\d.]+, (\d+), [\d.]+, [\w-]+ \[([\d. \n]+)]$', re.MULTILINE)
        o = os.path.join(*self.rundirectory_template[:-1], "optimization.progress.log")
        processed = 0
        if os.path.exists(o):
            with open(o, 'r') as progress_in:
                content = progress_in.read()
                results = matcher.findall(content)
                for (infected, x_probe) in results:
                    infected = int(infected)
                    x_probe = np.array([float(x) for x in re.split(r'\s+', x_probe) if x != ''])
                    minimizer.probed_X.append(x_probe)
                    minimizer.set_iota(x_probe, infected)
                    processed += 1

        return processed

    def leave_instructions(self, slave: int, x_probe: FloatArray, run: int):
        os.makedirs(self.instruction_dir, exist_ok=True)
        with open(os.path.join(self.instruction_dir, f"run-{slave}"), 'w') as instructions_out:
            if len(x_probe.shape) > 1:
                print("Ok, so our x_probe array is multi-dimensional, now we have a problem:", x_probe.shape)
            json.dump(dict(x_probe=list(x_probe), run=run, serialized=self.serialize_policy(x_probe)), instructions_out, indent=1)
            print(f"Created instructions for slave {slave}:", x_probe)

    def iterate_as_slave(self):
        instruction_file = os.path.join(self.instruction_dir, f"run-{self.slave_number}")
        progress_file = os.path.join(self.instruction_dir, f"run-{self.slave_number}.progress")
        done_file = os.path.join(self.instruction_dir, f"run-{self.slave_number}.done")

        while(True):
            while not os.path.exists(instruction_file):
                print(f"Slave {self.slave_number} is waiting for instructions")
                print(f"Waiting for file {instruction_file} to appear")
                time.sleep(30)

            with open(instruction_file, 'r') as instructions_in:
                instructions = json.load(instructions_in)
                print("Instructions found!")
            x_probe = np.asarray(instructions["x_probe"])
            run = instructions["run"]
            os.rename(instruction_file, progress_file)
            self.do_optimization_simulation(x_probe, run)
            os.rename(progress_file, done_file)

    @staticmethod
    def load_norm_weights(norm_weights_file) -> Dict[str, float]:
        norm_weights = dict()
        with open(norm_weights_file, 'r') as norm_weights_in:
            norm_weights_in.readline()[:-1].split("\t")  # Skip header
            for line in norm_weights_in:
                data = line.replace("\n", "").split("\t")
                norm_weights[data[0]] = float(data[1])
        return norm_weights

    def score_simulation_run(self, x: Dict[str, float], directories: List[Dict[int, str]]) -> float:
        target, infected, fitness = self.evaluator.fitness(directories, self.run_configuration['norm_schedule'])
        return target

    def _write_csv_log(self, score):
        pass

    @staticmethod
    def load_norm_application_counts_from_file(filename):
        norm_counts = dict()
        with open(filename, 'r') as file_in:
            file_in.readline()  # Skip header
            for line in file_in:
                data = line.split("\t")
                norm_counts[data[0]] = {'affected_agents': int(data[1]), 'affected_duration': int(data[2]), 'total_duration': int(data[2])}

        return norm_counts

    def x_to_norm_schedule(self, x: FloatArray):
        static, dynamic = self.digitize_x(x)
        schedule = self.create_norm_schedule(static, dynamic)
        return schedule

    def digitize_x(self, x: FloatArray) -> (IntArray, IntArray):
        static, dynamic = split_X(x, len(self.static_penalty), self.n_weeks)
        static_digitized = digitize_S(static, self.static_policy_nvals)
        dynamic_digitized = digitize_D(dynamic)
        return static_digitized, dynamic_digitized

    def create_norm_schedule(self, static: IntArray, dynamic: IntArray):
        """
        Creates a norm schedule object from the digitized array of static parameters, and the digitized dynamic
        parameter
        Args:
            static: Digitized array of static parameters
            dynamic:   Digitized dynamic parameter

        Returns:
            Norm schedule representing the parameter configuration proposed by the Bayesian optimizer
        """
        norm_matrix: Dict[str, List[str]] = dict()
        for norm, static_weeks in zip(sorted(self.policy['static']), static):
            norm_matrix[norm] = [self.policy['static'][norm][week]['value'] for week in static_weeks]
        norm_matrix['StayHomeSick'] = [self.policy['dynamic']['StayHomeSick'][week]['value'] for week in dynamic]

        schedule = NormSchedule.from_protocol_v3(
            norm_matrix,
            self.simulation_end,
            True
        )

        return schedule

    def serialize_policy(self, x: FloatArray):
        static, dynamic = self.digitize_x(x)
        return self.serialize_int_arrays(static, dynamic)

    def serialize_int_arrays(self, static_digitized: IntArray, dynamic_digitized: IntArray) -> str:
        max_int_size = max([len(norm) for norm in self.policy['static'].values()])
        bits = [utility.utility.int_list_to_int(norm, max_int_size) for norm in static_digitized]
        bits.append(utility.utility.int_list_to_int(dynamic_digitized, max_int_size))
        return "-".join(utility.utility.base_encode(b, 62) for b in bits)

    def deserialize_int_arrays(self, x: str):
        bits = [utility.utility.base_decode(_x, 62) for _x in x.split("-")]
        max_int_size = max([len(norm) for norm in self.policy['static'].values()])
        deserialized = [utility.utility.int_to_int_list(_x, max_int_size, self.n_weeks) for _x in bits]
        return deserialized[:-1], deserialized[-1]


