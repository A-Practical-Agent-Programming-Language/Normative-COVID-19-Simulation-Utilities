import math
import os
import re
from datetime import datetime

from classes.Epicurve_RMSE import Epicurve_RMSE
from classes.execution.CodeExecution import CodeExecution


class DiseaseCalibration(CodeExecution):
	rundirectory_template = ["disease", "{ncounties}counties-fips-{fips}", "{isymp}isymp-{iasymp}iasymp-{liberal}l-{conservative}c-{fatigue}f-{fatigue_start}fs-run{run}"]
	progress_format = "[DISEASE] [{time}] {ncounties} counties ({fips}): {score} for isymp {x[0]} and iasymp {x[1]} disease calibration (dir={output_dir})\n"
	csv_log = os.path.join("output", "calibration.disease.csv")

	def __init__(self, epicurve_rmse: Epicurve_RMSE, *args, **kwargs):
		super(DiseaseCalibration, self).__init__(*args, **kwargs)
		self.target_file = self.epicurve_filename
		self.base_disease_model = self.disease_model_file
		self.disease_model_file = os.path.join(
			'.persistent', '.tmp',
			f"scaled_disease_model_file_{kwargs['output_dir'].split(os.path.sep)[-1]}_" +
			f"{self.start_time.strftime('%Y_%m_%dT%H_%M_%S')}.toml")
		self.epicurve_rmse = epicurve_rmse
		self.scores = dict()

		exists = os.path.exists(self.csv_log)
		if not exists:
			with open(self.csv_log, "a") as fout:
				fout.write("fips,#counties,score,isymp,iasymp,time_finished,calibration_start_time\n")

	def calibrate(self, x):
		x[2] = math.ceil(x[2] / 2) * 2
		x = tuple(x)
		if x in self.scores:
			return self.scores[x]
		if 0 > x[1] > x[0] > 1 or x[2] > 50:
			return 999999999999
		else:
			score = super(DiseaseCalibration, self).calibrate(x)
			self.scores[x] = score
			return score

	def store_fitness_guess(self, x):
		self.run_configuration["isymp"] = x[0]
		self.run_configuration["iasymp"] = x[1]

	def prepare_simulation_run(self, x):
		self._scale_disease_model(x)

	def score_simulation_run(self, x):
		return self.epicurve_rmse.calculate_rmse(self.get_base_directory())

	def _scale_disease_model(self, x):
		with open(self.base_disease_model, 'r') as fin:
			with open(self.disease_model_file, 'w') as fout:
				for line in fin.readlines():
					if re.match(r'ia?symp.base\s*=\s*(\d\.)?\d+', line):
						if line.startswith("isymp.base"):
							fout.write(f"isymp.base = {x[0]}\n")
						elif line.startswith("iasymp.base"):
							fout.write(f"iasymp.base = {x[1]}\n")
					else:
						fout.write(line)

	def _write_csv_log(self, score):
		with open(self.csv_log, 'a') as fout:
			fout.write(
				"{fips},{ncounties},{score},{isymp},{iasymp},{finished_time},{starttime}\n".format(
					score=score,
					finished_time=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
					starttime=self.start_time.strftime("%Y-%m-%dT%H:%M:%S"),
					**self.run_configuration)
			)
