from classes.execution.CodeExecution import CodeExecution


class RepeatedExecution(CodeExecution):
	"""A simple class to repeatedly run an experiment with the same set of parameters a number of times"""

	rundirectory_template = ["repeat", "{ncounties}counties-fips-{fips}", "{liberal}l-{conservative}c-run{run}"]
	progress_format = "[REPEAT: {time}] {ncounties} counties ({fips}): {score} (dir={output_dir})\n"

	def __init__(self,  *args, **kwargs):
		super(RepeatedExecution, self).__init__(*args, **kwargs)

	def calibrate(self, x):
		super(RepeatedExecution, self).calibrate(x)
		print(f"Finished {self.n_runs} runs. Exiting")
		exit(0)

	def store_fitness_guess(self, x):
		pass

	def prepare_simulation_run(self, x):
		pass

	def score_simulation_run(self, x):
		return 0

	def _write_csv_log(self, score):
		pass
