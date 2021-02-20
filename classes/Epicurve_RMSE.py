import os
import re
from collections import defaultdict
from math import sqrt
from typing import Dict, Any
from sklearn.metrics import mean_squared_error


class Epicurve_RMSE(object):
	"""
	Class to calculate the RMSE between epicurve produced by the simulation with actual case data
	"""

	def __init__(
			self,
			counties: [Dict[str, Dict[str, Any]]],
			case_file: str = 'external/va-counties-covid19-cases.csv',
			scale_factor: int = 30
	):
		self.counties = counties
		self.case_file = case_file
		self.county_case_data = self.__load_case_data()
		self.baseline = self.__create_relevant_epicurve()
		self.scale_factor = scale_factor

	def __load_case_data(self):
		"""
		Load the case data available for the state
		Returns:
			Dictionary of dictionaries, where the outer key is the FIPS code of the county for which case data is
			known, the inner key is the date, and the value is the number of recorded cases until that day
		"""
		epicurve = dict()

		with open(self.case_file, 'r') as cases_in:
			headers = cases_in.readline()[:-1].split(",")
			for line in cases_in:
				data = line[:-1].split(",")
				date = data[headers.index("date")]
				fips_raw = data[headers.index("fips")]
				if fips_raw.lower() in ['', 'Unknown']:
					continue
				fips = int(re.match(r'(?:510?)?(\d{2,3})', fips_raw).groups()[0])
				cases = data[headers.index("cases")]

				if fips not in epicurve:
					epicurve[fips] = dict()
				epicurve[fips][date] = int(cases)

		return epicurve

	def __create_relevant_epicurve(self):
		"""
		Create one epicurve from the per-county case data, by summing the cases on each day for all counties
		participating in the simulation

		Returns:
			Dictionary with dates as key, and cumulative case counts for all counties in the simulation as value
		"""
		epicurve = defaultdict(int)
		for county in self.counties.values():
			if county["fipscode"] in self.county_case_data:
				for d in self.county_case_data[county["fipscode"]]:
					epicurve[d] += self.county_case_data[county["fipscode"]][d]

		if not len(epicurve):
			raise ValueError("No epicurve found for provided counties!")

		return epicurve

	def calculate_rmse(self, run_directory: str):
		"""
		Calculates the root mean squared error (RMSE) between the number of recovered agents in the simulation and
		the number of actually observed cases (the latter multiplied with {scale_factor} to account for testing
		uncertainty)

		Args:
			run_directory:    Output directory of simulation run

		Returns:
			Double: RMSE between scaled actual case count and number of agents recovered in the simulation

		"""
		predicted_recovered = self.__read_recovered_from_epicurve(run_directory)
		predicted, target = list(), list()

		dates = sorted(list(set([x for sublist in [list(predicted_recovered.keys()), list(self.baseline.keys())] for x in sublist])))

		# Calculate RSME
		for date in dates:
			if date in self.baseline and date in predicted_recovered:
				predicted.append(predicted_recovered[date])
				target.append(self.baseline[date] * self.scale_factor)

		# Write values used for calculating RSME to file, so plot of fits can be created later
		with open(os.path.join(run_directory, 'compare-case-data.csv'), 'w') as epicurve_out:
			epicurve_out.write("Date,Cases,ScaledCases,Recovered\n")
			for date in dates:
				cases = self.baseline[date] if date in self.baseline else ''
				scaled_cases = cases * self.scale_factor if cases != '' else ''
				recovered = predicted_recovered[date] if date in predicted_recovered else ''
				epicurve_out.write(f"{date},{cases},{scaled_cases},{recovered}\n")

		return sqrt(mean_squared_error(target, predicted))

	@staticmethod
	def __read_recovered_from_epicurve(run_directory: str):
		epicurve = dict()
		with open(os.path.join(run_directory, 'epicurve.sim2apl.csv'), 'r') as epicurve_in:
			headers = epicurve_in.readline()[:-1].split(";")
			for line in epicurve_in:
				data = line[:-1].split(";")
				epicurve[data[headers.index("Date")]] = int(data[int(headers.index("RECOVERED"))])

		return epicurve

