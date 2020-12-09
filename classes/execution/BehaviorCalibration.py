from classes.execution.CodeExecution import CodeExecution
from classes.Gyration import Gyration


class BehaviorCalibration(CodeExecution):
	regex = r"^(\w+\.base)\s*=\s*([0-9\.]+)$"

	def __init__(self, gyration: Gyration, tick_averages_file, *args, **kwargs):
		self.gyration = gyration
		self.target_file = tick_averages_file
		super(BehaviorCalibration, self).__init__(*args, **kwargs)

	def calibrate(self, x):
		if x[0] < 0 or x[0] > 1 or x[1] < 0 or x[1] > 1:
			return 999999999999
		else:
			super(BehaviorCalibration, self).calibrate(x)

	def store_fitness_guess(self, x):
		self.run_configuration["liberal"] = x[0]
		self.run_configuration["conservative"] = x[1]

	def prepare_simulation_run(self, x):
		"""Nothing to be done here"""
		pass

	def score_simulation_run(self, x):
		return self.gyration.calculate_rmse(self.get_base_directory())
