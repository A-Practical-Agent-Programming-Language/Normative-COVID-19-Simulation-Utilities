import os
from datetime import datetime

from classes.Gyration import Gyration
from classes.execution.CodeExecution import CodeExecution


class BehaviorCalibration(CodeExecution):
	regex = r"^(\w+\.base)\s*=\s*([0-9\.]+)$"
	csv_log = os.path.join("output", "calibration.behavior.csv")

	def __init__(self, gyration: Gyration, tick_averages_file, *args, **kwargs):
		self.gyration = gyration
		self.target_file = tick_averages_file
		super(BehaviorCalibration, self).__init__(*args, **kwargs)

		exists = os.path.exists(self.csv_log)
		if not exists:
			with open(self.csv_log, "a") as fout:
				fout.write("fips,#counties,score,liberal,conservative,time_finished,calibration_start_time\n")

	def calibrate(self, x):
		if x[0] < 0 or x[0] > 1 or x[1] < 0 or x[1] > 1:
			return 999999999999
		else:
			return super(BehaviorCalibration, self).calibrate(x)

	def store_fitness_guess(self, x):
		self.mode_liberal = x[0]
		self.mode_conservative = x[1]
		self.run_configuration["liberal"] = self.mode_liberal
		self.run_configuration["conservative"] = self.mode_conservative


	def prepare_simulation_run(self, x):
		"""Nothing to be done here"""
		pass

	def score_simulation_run(self, x):
		return self.gyration.calculate_rmse(self.get_base_directory())

	def _write_csv_log(self, score):
		with open(self.csv_log, 'a') as fout:
			fout.write("{fips},{ncounties},{score},{liberal},{conservative},{finished_time},{starttime}\n".format(
				score=score,
				finished_time=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
				starttime=self.start_time.strftime("%Y-%m-%dT%H:%M:%S"),
				**self.run_configuration)
			)
