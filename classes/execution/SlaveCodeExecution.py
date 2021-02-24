import json
import os
import time
from datetime import datetime

from classes.execution.CodeExecution import CodeExecution


class SlaveCodeExecution(CodeExecution):

	def __init__(self, run: int, *args, **kwargs):
		super(SlaveCodeExecution, self).__init__(*args, **kwargs)
		self.run = run
		self.__instruction_file = f".persistent/.tmp/{self.name}/run-{self.run}"
		self.__done_file = f".persistent/.tmp/{self.name}/run-{self.run}.DONE"
		self.global_name = self.name
		self.name = f".{self.name}.run{self.run}"

		self.initiate()

	def initiate(self):
		while True:
			# We want to continue running until we get DONE instructions

			while not os.path.exists(self.__instruction_file):
				# Waiting for instructions
				print("Waiting for instructions")
				time.sleep(1)

			self.start_run_time = datetime.now()
			instructions = open(self.__instruction_file, 'r').read()
			if instructions == "DONE":
				print("Received DONE status. Exiting")
				exit(0)
			self.run_configuration = json.loads(instructions)
			self.run_configuration["run"] = self.run
			os.remove(self.__instruction_file)

			if not os.path.exists(self.get_target_file()):
				# TODO, we can read the number of lines to see if the run was successful
				print(f"Starting run {self.run}")
				self.set_pansim_parameters()
				self.start_run() # TODO enable later
			else:
				print("Run already took place; skipping")

			with open(self.__done_file, 'w') as done_file:
				done_file.write(f"Run {self.run} finished")

	def calibrate(self, x):
		pass

	def store_fitness_guess(self, x):
		pass

	def prepare_simulation_run(self, x):
		pass

	def score_simulation_run(self, x):
		pass

	def _write_csv_log(self, score):
		pass
