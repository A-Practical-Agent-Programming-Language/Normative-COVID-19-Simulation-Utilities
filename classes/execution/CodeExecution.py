import json
import os
import random
import stat
import subprocess
import sys
import time
from abc import abstractmethod
from datetime import datetime

from classes.EssentialLocations import EssentialDesignationExtractor


class CodeExecution(object):
	epicurve_filename = "epicurve.csv"
	pansim_shell = ["time", "pansim", "simplesim"]
	# state_file: str

	"""Overridable members"""
	rundirectory_template = ["behavior", "{ncounties}counties-fips-{fips}", "{liberal}l-{conservative}c-{fatigue}f-{fatigue_start}fs-run{run}"]
	progress_format = "[BEHAVIOR: {time}] {ncounties} counties ({fips}): {score} for government attitudes liberal={x[0]}, conservative={x[1]}, fatigue={x[2]}, fatigue_start={x[3]} (dir={output_dir})\n"
	target_file = ""

	def __init__(
			self,
			county_configuration_file,
			sim2apl_jar,
			counties,
			disease_model_file,
			n_cpus=1,
			java_home="java",
			java_threads=1,
			java_heap_size_max="64g",
			java_heap_size_initial="55g",
			output_dir=None,
			mode_liberal=0.5,
			mode_conservative=0.5,
			fatigue=0.0125,
			fatigue_start=60,
			n_runs=10,
			epicurve_file="epicurve.csv",
			name=None,
			is_master=False
	):
		self.county_configuration_file = county_configuration_file
		self.counties = counties
		self.n_cpus = n_cpus
		self.java_home = java_home
		self.java_threads = java_threads
		self.max_heap_size = java_heap_size_max
		self.initial_heap_size = java_heap_size_initial
		self.sim2apl_jar = sim2apl_jar
		self.epicurve_filename = epicurve_file
		self.n_runs = n_runs
		self.rundirectory_template.insert(0, output_dir)

		self.mode_liberal = mode_liberal
		self.mode_conservative = mode_conservative
		self.fatigue = fatigue
		self.fatigue_start = fatigue_start

		self.disease_model_file = disease_model_file

		self.name = name
		self.is_master = is_master

		# self.state_file = StateFile(self.counties).merge_from_config()

		if not os.path.exists("output"):
			os.makedirs("output")

		if n_cpus > 1:
			self.pansim_shell = self._make_distributed_run(n_cpus)

		self.counties = counties
		self.run_configuration = dict(
			fips="-".join(sorted(map(lambda x: str(x["fipscode"]), counties.values()))),
			ncounties=len(counties),
			liberal=self.mode_liberal,
			conservative=self.mode_conservative,
			fatigue=self.fatigue,
			fatigue_start=self.fatigue_start
		)

		self.lid_partition, self.pid_partition = self.get_partitions()
		self.start_time = datetime.now()

	def get_partitions(self) -> (str, str):
		fname = f"_partition_{self.n_cpus}"
		dir_name = f"partitions_{self.run_configuration['fips']}"
		lid_partition = os.path.join('.persistent', dir_name, "lid" + fname)
		pid_partition = os.path.join('.persistent', dir_name, "pid" + fname)
		locations = ["pansim", "partition", "-l", lid_partition, "-p", pid_partition, "-n", str(1), "-c", str(self.n_cpus)]
		for conf in self.counties.values():
			locations.extend(conf["activities"] if not "locations" in conf else conf["locations"])
		if not os.path.exists(lid_partition) or not os.path.exists(pid_partition):
			os.makedirs(os.path.join('.persistent', dir_name), exist_ok=True)
			subprocess.run(locations)
		return lid_partition, pid_partition

	def get_base_directory(self, filename=None):
		t = list(map(lambda x: x.format(**self.run_configuration), self.rundirectory_template))
		if filename is not None:
			t.append(filename)
		return os.path.join(*t)

	def get_target_file(self):
		return self.get_base_directory(self.target_file)

	def calibrate(self, x):
		scores = list()
		self.store_fitness_guess(x)
		print("Finding loss for ", x)
		self.start_batch_time = datetime.now()

		for i in range(self.n_runs):
			self.run_configuration["run"] = i
			self.start_run_time = datetime.now()

			self.prepare_simulation_run(x)

			if self.is_master and i < self.n_runs - 1:
				# Leave instructions for slave node
				os.makedirs(f".persistent/.tmp/{self.name}", exist_ok=True)
				with open(f".persistent/.tmp/{self.name}/run-{i}", 'w') as instructions:
					self.run_configuration["run_directory_template"] = self.rundirectory_template
					self.run_configuration["disease_model_file"] = self.disease_model_file
					self.run_configuration["county_configuration_file"] = self.county_configuration_file
					instructions.write(json.dumps(self.run_configuration))
			else:
				if not os.path.exists(self.get_target_file()):
					print("Starting run ", self.run_configuration["run"], "See tail -f calibration.run.log for progress")
					java_command_file = self._create_java_command_file()
					self.set_pansim_parameters(java_command_file)
					self.start_run(java_command_file)
					os.remove(java_command_file)
				else:
					print(
						"\nRun", self.run_configuration["run"],
						"already took place; skipping. Delete the directory for that run if it needs to be calculated again")

			if self.is_master and i >= self.n_runs - 1:
				while not self.__all_runs_finished():
					print("Waiting for other runs to finish")
					time.sleep(1)

				for j in range(self.n_runs):
					self.run_configuration["run"] = j
					print("Calculating loss for " + self.get_target_file())
					scores.append(self.score_simulation_run(x))
			elif not self.is_master:
				print("Calculating loss for " + self.get_target_file())
				scores.append(self.score_simulation_run(x))

		return self._process_loss(x, scores)

	def __all_runs_finished(self):
		if not self.is_master:
			return True

		for i in range(self.n_runs - 1):
			if not os.path.exists(f".persistent/.tmp/{self.name}/run-{i}.DONE"):
				return False

		for i in range(self.n_runs - 1):
			if os.path.exists(f".persistent/.tmp/{self.name}/run-{i}.DONE"):
				os.remove(f".persistent/.tmp/{self.name}/run-{i}.DONE")

		return True

	def _process_loss(self, x, scores):
		loss = sum(scores) / len(scores)
		print("Loss for", x, "is", loss)

		args = dict(
			x=x,
			run=self.run_configuration,
			ncounties=self.run_configuration["ncounties"],
			fips=self.run_configuration["fips"],
			score=loss,
			time=time.ctime(),
			output_dir=self.get_base_directory()
		)
		with open(os.path.join("output", "calibration.results.log"), "a") as fout:
			fout.write(self.progress_format.format(**args))

		self._write_csv_log(loss)
		return loss

	def set_pansim_parameters(self, java_command_file):
		home = os.environ["HOME"]
		path = os.environ["PATH"]
		os.environ.clear()
		os.environ["HOME"] = home
		os.environ["PATH"] = path
		os.environ["XACTOR_MAX_SEND_BUFFERS"] = str(4 * self.n_cpus)
		os.environ["XACTOR_MAX_MESSAGE_SIZE"] = str(33554432)
		os.environ["OUTPUT_FILE"] = self.get_base_directory(self.epicurve_filename)
		os.environ["SEED"] = str(random.randrange(sys.maxsize))
		os.environ["DISEASE_MODEL_FILE"] = self.disease_model_file
		os.environ["TICK_TIME"] = str(1)
		os.environ["NUM_TICKS"] = str(120)
		os.environ["MAX_VISITS"] = str(204000)
		os.environ["VISUAL_ATTRIBUTES"] = "coughing,mask,sdist"
		os.environ["LID_PARTITION"] = self.lid_partition
		os.environ["PID_PARTITION"] = self.pid_partition
		os.environ["PER_NODE_BEHAVIOR"] = str(1)
		os.environ["JAVA_BEHAVIOR"] = str(1)
		os.environ["JAVA_BEHAVIOR_SCRIPT"] = java_command_file
		os.environ["TIMEFORMAT"] = "Simulation runtime: %E"
		# os.environ["START_STATE_FILE"] = os.path.abspath(self.state_file)

	@abstractmethod
	def store_fitness_guess(self, x):
		pass

	@abstractmethod
	def prepare_simulation_run(self, x):
		pass

	@abstractmethod
	def score_simulation_run(self, x):
		pass

	@staticmethod
	def _make_distributed_run(N_CPUS):
		"""Change pansim arguments if distributed version (distsim) should be used"""
		return ["time", "mpiexec", "--mca", "mpi_yield_when_idle", "1", "-n", str(N_CPUS), "pansim", "distsim"]

	@abstractmethod
	def _write_csv_log(self, score):
		pass

	def start_run(self, java_command_file):
		"""
		Runs and monitors one iteration of the integrated simulation
		Closes the Java process if something goes wrong
		"""
		pansim_process = self._start_pansim_process()
		if pansim_process.returncode != 0:
			print(f"Failed to complete simulation. Pansim status {pansim_process.returncode}")
			os.remove(java_command_file)
			exit(pansim_process.returncode)

	def _create_java_command_file(self, suffix: str = None):
		fname = os.path.abspath(os.path.join(".persistent", ".tmp", f"start_behavior_model_{time.time()}"))
		if suffix is not None:
			fname += suffix
		fname += ".sh"
		with open(fname, 'w') as command_out:
			command_out.write("#!/bin/bash\n\n")
			command_out.write(" ".join(self._java_command()))
		state = os.stat(fname)
		os.chmod(fname, state.st_mode | stat.S_IEXEC)
		return fname

	def __get_agent_run_log_file(self):
		name = self.name if self.name is None or self.name.startswith(".") else "." + self.name
		name = "" if name is None else name
		name = name.replace(" ", "_")
		if self.is_master:
			name += ".master"
		agent_run_log_file_name = f"calibration.agents{name}.{self.start_time.strftime('%Y_%m_%dT%H_%M_%S')}.run.log"
		return os.path.abspath(os.path.join(self.get_base_directory(), agent_run_log_file_name))

	def _start_pansim_process(self):
		"""Starts PanSim"""
		return subprocess.run(self.pansim_shell)

	def _java_command(self):
		return [
			self.java_home,
			f"-Xmx{self.max_heap_size}",
			f"-Xms{self.initial_heap_size}",
			"-XX:+UseNUMA", "-jar",
			os.path.abspath(self.sim2apl_jar),
			"--config", os.path.abspath(self.county_configuration_file),
			"--mode-liberal", str(self.mode_liberal),
			"--mode-conservative", str(self.mode_conservative),
			"--fatigue", str(self.fatigue),
			"--fatigue-start", str(int(self.fatigue_start)),
			"-t", str(self.java_threads),
			"-c",
			"--output", self.get_base_directory()
		] + self.get_extra_java_commands() + [
			">", self.__get_agent_run_log_file(), "2>&1"
		]

	def get_extra_java_commands(self):
		"""Allows implementing sub-classes to specify additional parameters for the Java behavior model"""
		return []
