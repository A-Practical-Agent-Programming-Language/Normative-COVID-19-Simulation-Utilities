import os

from classes.Gyration import Gyration
from classes.execution.BehaviorCalibration import BehaviorCalibration


class SensitivityAnalysis(BehaviorCalibration):
	"""
	This class is used to test the sensitivity of agent behavior to trust attitude, and runs
	the simulation for {number_of_runs} times for the following trust parameters:
	(0,0)
	(1,1)
	(0,1)
	(1,0)
	(0.5,0.5)
	"""
	rundirectory_template = ["sensitivity", "{ncounties}counties-fips-{fips}", "{liberal}l-{conservative}c-{fatigue}f-{fatigue_start}fs-run{run}"]
	csv_log = os.path.join("output", "sensitivity.behavior.csv")

	def __init__(self, gyration: Gyration, tick_averages_file: str, *args, **kwargs):
		super(SensitivityAnalysis, self).__init__(gyration, tick_averages_file, *args, **kwargs)
		for x in [[0, 0], [1, 1], [0, 1], [1, 0], [.5, .5]]:
			params = [x[0], x[1], 0.125, 60]
			print("Running simulation for ", params)
			super(SensitivityAnalysis, self).calibrate(params)

		print("Finished")

	def prepare_simulation_run(self, x):
		if x[0] not in [0, 1, 0.5] or x[1] not in [0, 1, 0.5]:
			print("Finished")
			exit(1)
