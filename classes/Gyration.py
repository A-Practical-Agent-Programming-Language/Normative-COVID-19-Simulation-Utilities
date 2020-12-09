import datetime
import os
import subprocess
from math import sqrt
from typing import List, Dict, Any
from scipy.optimize import minimize


class Gyration(object):

	def __init__(self, va_gyration_mobility_index_file: str, tick_averages_file_name: str, counties: [Dict[str, Dict[str, Any]]]) -> None:
		self.__day_average_file_template = os.path.join('.persistent', 'day-averages', 'day-averages-{0}.csv')
		self._va_gyration_mobility_index_file = va_gyration_mobility_index_file
		self._baseline_mobility_index = self._load_baseline_mobility_index()
		self._tick_averages_file_name = tick_averages_file_name
		self.__day_baselines = dict()

		for county in counties.values():
			try:
				self._get_baseline_for_county(county["fipscode"], county["activities"])
			except KeyError:
				self._get_baseline_for_county(county["fipscode"], county["locations"])

	def _get_baseline_for_county(self, fips_code: int, activity_files: List[str] = None) -> Dict[int, float]:
		"""
		Calculates or retrieves the default weekly radius of gyration for a county based on the activity files.
		This can be used to calculate the mobility index produced by the behavior model
		Args:
			activity_files: All activity files containing location assignments for the specified county
			fips_code:   FIPS code of the specified county

		Returns:
			Dictionary with keys ranging from 0 (Monday) to 6 (Sunday) and values the radius of gyration for that day

		"""
		if fips_code in self.__day_baselines:
			return self.__day_baselines[fips_code]
		else:
			stored_day_averages = self.__day_average_file_template.format(fips_code)
			baseline = self._load_county_baseline_radius_from_file(stored_day_averages)

			if baseline is None and activity_files is not None:
				print("Baseline for county {0} not yet present. Calculating...".format(fips_code))
				self._calculate_baseline_radius_of_gyration(activity_files, fips_code)

			baseline = self._load_county_baseline_radius_from_file(stored_day_averages)

			self.__day_baselines[fips_code] = baseline
		return baseline

	@staticmethod
	def _load_county_baseline_radius_from_file(stored_day_averages: str) -> Dict[int, float] or None:
		if os.path.exists(stored_day_averages):
			day_averages = dict()
			with open(stored_day_averages, 'r') as fin:
				for f in fin.readlines():
					line = f.split(",")
					day_averages[int(line[0])] = float(line[1])
			for i in range(6):
				if i not in day_averages:
					return None

			return day_averages

	def _calculate_baseline_radius_of_gyration(self, activity_files: List[str], fips_code: int) -> Dict[int, float]:
		"""
		Calculate the baseline for the radius of gyration for the activity files of a county, by running the calibration
		script on each day in the activity file (7 days in total, 0 being monday, 6 being sunday) and write the
		results to file
		"""

		locations_dir = os.path.join("external", "locations_{0}".format(fips_code))
		output_dir = os.path.join("external", "output_{0}".format(fips_code))

		self.__cleanup_visit_tmp_data(locations_dir, output_dir, fips_code)
		os.makedirs(locations_dir)

		# Keeps track of activities per day
		locations_per_day = self._map_visits_to_days(activity_files)

		# Write each set of locations to a file corresponding to that day
		for c in locations_per_day:
			f_name = os.path.join(locations_dir, f'day-{c}-locations.csv')
			with open(f_name, 'w') as f:
				for line in locations_per_day[c]:
					f.write(line)

		# Call gyration calculation script (uses tab as separator)
		subprocess.run(['python3', os.path.join("external", 'gyration_radius_calculator.py'), '-i', locations_dir, '-o', output_dir])

		# Calculate averages for each day
		average_per_day = self._process_gyration_radius_calculator_output(output_dir)
		self.__cleanup_visit_tmp_data(locations_dir, output_dir, fips_code)
		self._store_day_averages(average_per_day, fips_code)

		return average_per_day

	@staticmethod
	def _read_tick_averages_file(tick_averages_file: str) -> Dict[int, Dict[str, float]]:
		tick_values = dict()
		with open(tick_averages_file, 'r') as in_file:
			for line in in_file.read().splitlines():
				date, fips, radius = line.split(",")
				fips = int(fips)
				radius = float(radius)
				if fips not in tick_values:
					tick_values[fips] = dict()
				tick_values[fips][date] = radius

		return tick_values

	def calculate_rmse(self, run_directory):
		"""
		Calculate the Root Mean Square Error (RMSE) of a simulation run, by comparing the
		mobility index (percentage change from the baseline for each day) of the agents to
		that observed on the same day in real life
		"""
		tick_averages = self._read_tick_averages_file(os.path.join(run_directory, self._tick_averages_file_name))
		pct_reduction_predicted, pct_reduction_target, overview = self._make_mobility_index_comparison_lists(tick_averages)

		self._write_tick_averages_to_run_directory(sorted(tick_averages.keys()), overview, run_directory)

		# Calculate the RMSE by comparing the actual vs. simulated mobility indices
		return sqrt(mean_squared_error(pct_reduction_target, pct_reduction_predicted))

	@staticmethod
	def _write_tick_averages_to_run_directory(fips_codes: List[int], overview: Dict[str, Dict[int, Dict[str, float]]], run_directory: str) -> None:
		with open(os.path.join(run_directory, "mobility_index.csv"), 'w') as out:
			out.write(",".join(["date"] + sorted(list(map(lambda x: str(x), fips_codes))*2)))
			out.write(",".join(["\n"] + [x for _ in fips_codes for x in ["real", "agents", "agent_radiusKM"]]) + "\n")

			for d in sorted(overview.keys()):
				line_values = [d] + [str(overview[d][fips][x]) for fips in fips_codes for x in ["real", "agents", "agent_radius"]]
				out.write(",".join(line_values) + "\n")

	@staticmethod
	def _try_get_radius_from_dict(fips: int, date: str, dct: Dict[int, Dict[str, float]]) -> str:
		return dct[fips][date] if fips in dct and date in dct[fips] else ''

	def _make_mobility_index_comparison_lists(self, tick_averages: Dict[int, Dict[str, float]]) -> (List[float], List[float], Dict[str, Dict[int, Dict[str, float]]]):
		"""Calculate and compare the mobility index for each day to the baseline (if present)"""
		pct_reduction_predicted, pct_reduction_target = list(), list()
		mobility_index_overview = dict()

		for fips in tick_averages.keys():
			day_baseline = self._get_baseline_for_county(fips)

			for date in sorted(tick_averages[fips].keys()):
				if date not in mobility_index_overview:
					mobility_index_overview[date] = dict()
				dayofweek = datetime.date(*list(map(lambda x: int(x), date.split("-")))).weekday()
				percent_reduction = (tick_averages[fips][date] - day_baseline[dayofweek]) / day_baseline[dayofweek] * 100.0
				mobility_index_overview[date][fips] = dict(real="", agents=percent_reduction, agent_radius=tick_averages[fips][date])
				if date in self._baseline_mobility_index[fips]:
					mobility_index_overview[date][fips]["real"] = self._baseline_mobility_index[fips][date]
					pct_reduction_predicted.append(percent_reduction)
					pct_reduction_target.append(self._baseline_mobility_index[fips][date])
				else:
					print("Missing date {0} for county FIPS {1} in va_baseline".format(date, fips))

		return pct_reduction_predicted, pct_reduction_target, mobility_index_overview

	@staticmethod
	def _start_time_to_day(start_time: int or str) -> int:
		"""Convert a start time (seconds since sunday midnight) to a day of week"""
		return int(int(start_time) / 60 / 60 / 24)

	@staticmethod
	def _remove_dir(dir_to_remove: str) -> None:
		if os.path.exists(dir_to_remove):
			for f in os.listdir(dir_to_remove):
				os.remove(dir_to_remove + '/' + f)
			os.rmdir(dir_to_remove)

	def _load_baseline_mobility_index(self) -> Dict[int, Dict[str, float]]:
		"""
		Loads the baseline mobility data (mobility index calculated on cubric data) as
		a dictionary with the county fips code as the key
		"""
		with open(self._va_gyration_mobility_index_file, 'r') as baseline:
			lines = baseline.readlines()
			headers = lines[0].replace("\n", "").split(",")[1:]
			baseline = dict([(int(x), dict()) for x in headers])

			for line in lines[1:]:
				l_split = line.replace("\n", "").split(",")
				l_dict = dict(zip(headers, list(map(lambda x: float(x), l_split[1:]))))

				for i, h in enumerate(headers):
					baseline[int(h)][l_split[0]] = l_dict[h]

		return baseline

	def _store_day_averages(self, day_averages, fips_code):
		"""
		Store the calculated day averages in a dedicated file for the county, so it can be easily accessed later
		"""
		if not os.path.exists(os.path.dirname(self.__day_average_file_template)):
			os.makedirs(os.path.dirname(self.__day_average_file_template))
		with open(self.__day_average_file_template.format(fips_code), 'w') as fout:
			for k in day_averages:
				fout.write("{0},{1}\n".format(k, day_averages[k]))

	@staticmethod
	def _process_gyration_radius_calculator_output(output_dir: str) -> Dict[int, float]:
		average_per_day = dict()
		for f in os.listdir(output_dir):
			day = int(f[4])
			kms = []
			with open(output_dir + '/' + f) as fl:
				for line in fl.readlines()[1:]:
					kms.append(float(line.split(",")[-1].replace("\n", "")))
			average_per_day[day] = sum(kms) / len(kms)

		return average_per_day

	@staticmethod
	def _map_visits_to_days(activity_files: List[str]):
		"""Read passed locations file, and add each location file to the locations dictionary for the corresponding day"""
		locations = dict(zip(range(7), [[] for _ in range(7)]))

		for activity_file in activity_files:
			with open(activity_file, 'r') as f:
				lines = f.read().splitlines()
				headers = lines[0].split(",")
				for line in lines[1:]:
					cols = dict(zip(headers, line.split(",")))
					# Only latitude and longitude are required by the radius_of_gyration script, but these are stored in
					# columns 3 and 4 in the files that script expects
					# Note that we switch around col 7 (latitude) and 8 (longitude)
					day = Gyration._start_time_to_day(cols["start_time"])
					locations[day].append("\t".join([cols[i] for i in ["hid", "pid", "lid", "latitude", "longitude"]]) + "\n")

		return locations

	def __cleanup_visit_tmp_data(self, locations_dir: str, output_dir: str, fips_code: int) -> None:
		self._remove_dir(locations_dir)
		self._remove_dir(output_dir)
		if os.path.exists(os.path.join("external", "gyration_radius_calculator_locations_{0}.log".format(fips_code))):
			os.remove(os.path.join("external", "gyration_radius_calculator_locations_{0}.log".format(fips_code)))
