from typing import List, Dict

from classes.execution.CodeExecution import CodeExecution


class ScalingTestExecution(CodeExecution):

	target_file = "timings.csv"
	rundirectory_template = ["scaling", "{ncounties}counties-fips-{fips}", "{threads}", "run{run}"]
	progress_format = "[SCALING: {time}] {ncounties} counties ({fips}): {score} (dir={output_dir})\n"

	def __init__(self, suppress_calculations, *args, **kwargs):
		super(ScalingTestExecution, self).__init__(*args, **kwargs)
		self.suppress_calculations = suppress_calculations

	def store_fitness_guess(self, x):
		self.run_configuration["threads"] = x
		self.java_threads = x
		self.n_cpus = x
		self.pansim_shell = self._make_distributed_run(x)
		self.lid_partition, self.pid_partition = self.get_partitions()

	def prepare_simulation_run(self, x):
		pass

	def score_simulation_run(self, x, directories: List[Dict[int, str]]) -> float:
		pass

	def _write_csv_log(self, score):
		pass

	def _process_loss(self, x, loss):
		pass

	def get_extra_java_commands(self):
		return ["--suppress-calculations"] if self.suppress_calculations else []