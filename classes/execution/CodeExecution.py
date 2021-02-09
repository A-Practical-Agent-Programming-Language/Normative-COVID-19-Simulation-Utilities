import os
import random
import subprocess
import sys
import time
from abc import abstractmethod
from datetime import datetime


class CodeExecution(object):
	epicurve_filename = "epicurve.csv"
	pansim_shell = ["time", "pansim", "simplesim"]

	"""Overridable members"""
	rundirectory_template = ["behavior", "{ncounties}counties-fips-{fips}", "{liberal}l-{conservative}c-run{run}"]
	progress_format = "[BEHAVIOR: {time}] {ncounties} counties ({fips}): {score} for government attitudes liberal={x[0]}, conservative={x[1]} (dir={output_dir})\n"
	target_file = ""

	def __init__(self, county_configuration_file, sim2apl_jar, counties, disease_model_file, n_cpus=1, java_home="java", java_threads=1, java_heap_size_max="64g", java_heap_size_initial="55g", output_dir=None, mode_liberal=0.5, mode_conservative=0.5, n_runs=10, epicurve_file="epicurve.csv"):
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

		self.disease_model_file = disease_model_file

		if not os.path.exists("output"):
			os.makedirs("output")

		if n_cpus > 1:
			self.pansim_shell = self._make_distributed_run(n_cpus)

		self.counties = counties
		self.run_configuration = dict(
			fips="-".join(sorted(map(lambda x: str(x["fipscode"]), counties.values()))),
			ncounties=len(counties)
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

			# AgentState().scramble_state("shuffled_start_state.csv")
			self.prepare_simulation_run(x)

			if not os.path.exists(self.get_target_file()):
				print("Starting run ", self.run_configuration["run"], "See tail -f calibration.run.log for progress")
				self.set_pansim_parameters()
				self.start_run()
			else:
				print(
					"\nRun", self.run_configuration["run"],
					"already took place; skipping. Delete the directory for that run if it needs to be calculated again")

			print("Calculating loss for " + self.get_target_file())
			scores.append(self.score_simulation_run(x))

		return self._process_loss(x, scores)

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

	def set_pansim_parameters(self):
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
		os.environ["TIMEFORMAT"] = "Simulation runtime: %E"

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

	def start_run(self):
		"""
		Runs and monitors one iteration of the integrated simulation
		Closes the Java process if something goes wrong
		"""
		java_process = self._start_java_background_process()
		pansim_process = self._start_pansim_process()
		if pansim_process.returncode != 0 or (java_process.returncode is not None and java_process.returncode != 0):
			print(f"Failed to complete simulation. Pansim status {pansim_process.returncode}, java status {java_process.returncode}")
			java_process.kill()
			exit(pansim_process.returncode)

	def _start_java_background_process(self):
		"""Executes the Java 2APL behavior model in the background"""
		agentrun_log = os.path.join("output", f"calibration.agents.{self.start_time.strftime('%Y_%m_%dT%H_%M_%S')}.run.log")
		with open(agentrun_log, "a") as logfile:
			return subprocess.Popen(self._java_command(), stdout=logfile, stderr=logfile)

	def _start_pansim_process(self):
		"""Starts PanSim"""
		return subprocess.run(self.pansim_shell)

	def _java_command(self):
		return [
			self.java_home, f"-Xmx{self.max_heap_size}", f"-Xms{self.initial_heap_size}", "-jar", self.sim2apl_jar, "--config", self.county_configuration_file,
			"--mode-liberal", str(self.mode_liberal), "--mode-conservative", str(self.mode_conservative), "-t", str(self.java_threads), "-c",
			"--output", self.get_base_directory()
		] + self.get_extra_java_commands()

	def get_extra_java_commands(self):
		"""Allows implementing sub-classes to specify additional parameters for the Java behavior model"""
		return []
