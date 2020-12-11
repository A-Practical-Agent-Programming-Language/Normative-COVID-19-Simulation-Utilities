import os
import re
from datetime import datetime

from classes.Epicurve import Epicurve
from classes.execution.CodeExecution import CodeExecution


class DiseaseCalibration(CodeExecution):
	regex = r"^(\w+\.base)\s*=\s*([0-9\.]+)$"
	rundirectory_template = ["disease", "{ncounties}counties-fips-{fips}", "{scale}scale-run{run}-{liberal}l-{conservative}c"]
	progress_format = "[DISEASE] [{time}] {ncounties} counties ({fips}): {score} for scale factor {x[0]} seeding x[2] agents for x[1] days in disease calibration (dir={output_dir})"
	csv_log = os.path.join("output", "calibration.disease.csv")

	def __init__(self, *args, **kwargs):
		super(DiseaseCalibration, self).__init__(*args, **kwargs)
		self.target_file = self.epicurve_filename
		self.base_disease_model = self.disease_model_file
		self.disease_model_file = os.path.join('.persistent', '.tmp', "scaled_disease_model_file.toml")

		self.run_configuration["liberal"] = self.mode_liberal
		self.run_configuration["conservative"] = self.mode_conservative

		exists = os.path.exists(self.csv_log)
		if not exists:
			with open(self.csv_log, "a") as fout:
				fout.write("score,scale,n_days,n_agents_per_day,time_finished,calibration_start_time\n")

	def calibrate(self, x):
		if x[1] < 0 or x[2] < 0:
			return 999999999999
		else:
			return super(DiseaseCalibration, self).calibrate(x)

	def store_fitness_guess(self, x):
		self.run_configuration["scale"] = x[0]
		self.run_configuration["agents-per-day-seeding"] = int(x[1] * 100)
		self.run_configuration["days-seeding"] = int(x[2] * 100)

	def prepare_simulation_run(self, x):
		self._scale_disease_model(x[0])

	def score_simulation_run(self, x):
		return Epicurve(self.get_target_file()).get_score()

	def _scale_disease_model(self, scale):
		with open(self.base_disease_model, 'r') as fin:
			with open(self.disease_model_file, 'w') as fout:
				for line in fin.readlines():
					match = re.match(self.regex, line)
					if match:
						fout.write("{0} = {1}\n".format(match.group(1), scale * float(match.group(2))))
					else:
						fout.write(line)

	def get_extra_java_commands(self):
		return ["--disease-seed-days", str(self.run_configuration["days-seeding"]), "--disease-seed-number", str(self.run_configuration["agents-per-day-seeding"])]

	def _write_csv_log(self, score):
		with open(self.csv_log, 'a') as fout:
			fout.write("{score},{scale},{days-seeding},{agents-per-day-seeding},{finished_time},{starttime}\n".format(score=score,finished_time=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), starttime=self.start_time, **self.run_configuration))