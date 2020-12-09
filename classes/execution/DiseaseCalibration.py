import os
import re

from classes.execution.CodeExecution import CodeExecution
from classes.execution.Epicurve import Epicurve


class DiseaseCalibration(CodeExecution):
	regex = r"^(\w+\.base)\s*=\s*([0-9\.]+)$"
	rundirectory_template = ["scaling", "{ncounties}counties-fips-{fips}", "{scale}scale-run{run}-{liberal}l-{conservative}c"]
	progress_format = "[DISEASE] [{time}] {ncounties} counties ({county['fips']}): {score} for scale factor {x[0]} in disease calibration (dir={output_dir})"

	def __init__(self, *args, **kwargs):
		super(DiseaseCalibration, self).__init__(*args, **kwargs)
		self.target_file = self.epicurve_filename
		self.base_disease_model = self.disease_model_file
		self.disease_model_file = os.path.join('.persistent', '.tmp', "scaled_disease_model_file.toml")

		self.run_configuration["liberal"] = self.mode_liberal
		self.run_configuration["conservative"] = self.mode_conservative

	def store_fitness_guess(self, x):
		self.run_configuration["scale"] = x[0]

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
