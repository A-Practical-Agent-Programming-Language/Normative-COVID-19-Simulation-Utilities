import sys
import os
from collections import defaultdict

from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np


class EssentialDesignationExtractor(object):

	synthetic_population_location: str
	synth_pop_files: Dict[str, List[str]]
	essential_workers: List[str]
	num_total_agents: int
	num_essential_agents: int
	home_locations: Dict[str, bool]
	essential_locations: Dict[str, str]

	def from_synthetic_population_directory(self, synthetic_population_location: str):
		self.synthetic_population_location = synthetic_population_location
		self.synth_pop_files = self.find_files()
		self.__extract_for_county(os.path.join(self.synthetic_population_location, 'location_designation.csv'))

	def from_county(self, county: dict):
		self.synth_pop_files = {
			"activities": county["locations"] if "locations" in county else county["activities"],
			"household": county["households"],
			"person": county["persons"]
		}

		return self.__extract_for_county(self.synth_pop_files["household"][0].replace("household", "essential_location_designation"))

	def __extract_for_county(self, output_file):
		self.essential_workers, self.num_total_agents, self.num_essential_agents = self.load_essential_workers()
		self.home_locations = self.load_home_locations()
		self.essential_locations = self.add_residences_to_location_designations()
		self.mark_government_designation_with_highest_num_unique_visits_as_dmv()
		return self.write_locations_file(output_file)

	def find_files(self) -> Dict[str, List[str]]:
		"""
		Find all the files associated with a specific county
		"""
		synth_pop_files = {
			'person': list(),
			'activities': list(),
			'household': list()
		}

		for f in os.listdir(self.synthetic_population_location):
			if f.endswith('.csv'):
				if 'location' in f and 'designation' not in f or 'activit' in f:
					synth_pop_files['activities'].append(os.path.join(self.synthetic_population_location, f))
				elif 'household' in f:
					synth_pop_files['household'].append(os.path.join(self.synthetic_population_location, f))
				elif 'person' in f:
					synth_pop_files['person'].append(os.path.join(self.synthetic_population_location, f))

		# Deal with old style, where locations and activities were tracked in separate files
		if 'location' in "".join(synth_pop_files['activities']):
			synth_pop_files['activities'] = [act_file for act_file in synth_pop_files['activities'] if 'location' in act_file]

		return synth_pop_files

	def load_essential_workers(self) -> (Dict[str, str], int, int):
		"""
		From the person files of the synthetic population, extract those with an essential worker designation
		"""
		essentials = dict()
		total = essential = 0
		for f in self.synth_pop_files['person']:
			with open(f, 'r') as people:
				headers = people.readline()[:-1].split(",")
				pid_index = headers.index('pid')
				designation_index = headers.index('designation')
				for person in people:
					data = person[:-1].split(",")
					if data[designation_index] != 'none':
						essentials[data[pid_index]] = data[designation_index]
						essential += 1
					total += 1

		return essentials, total, essential

	def load_home_locations(self) -> Dict[str, bool]:
		homes = defaultdict(bool)
		for f in self.synth_pop_files['household']:
			with open(f, 'r') as homes_in:
				headers = homes_in.readline()[:-1].split(",")
				residence_id = headers.index('rlid')
				business_index = headers.index('business')
				for line in homes_in:
					home = line[:-1].split(",")
					homes[home[residence_id]] = (home[business_index] == "1" or homes[home[residence_id]])

		return homes

	def _assign_essential_locations(self) -> Dict[str, Set[str]]:
		"""
		Assign the LID's corresponding to the work location of agents with an essential worker designation
		as an essential location
		"""
		locations = defaultdict(set)
		for f in self.synth_pop_files['activities']:
			with open(f, 'r') as locations_in:
				headers = locations_in.readline()[:-1].split(",")
				try:
					pid_index = headers.index('pid')
				except Exception as e:
					print(e)
				type_index = headers.index('activity_type')
				lid_index = headers.index('lid')
				for location in locations_in:
					data = location[:-1].split(",")
					if data[pid_index] in self.essential_workers and data[type_index] == "2":
						locations[data[pid_index]].add(data[lid_index])

		double_work_locations = [loc for loc in locations if len(locations[loc]) > 1]
		if len(double_work_locations):
			print("WARNING: The following essential worker PID's have a different number of work locations than 1:")
			print(double_work_locations)

		return locations

	def write_locations_file(self, output_file):
		if os.path.exists(output_file):
			print(output_file, "already exists. Please remove this file to create it again. Reusing for now")
			return output_file
		with open(output_file, 'w') as out:
			out.write("lid,designation,isResidential\n")
			for loc, designation in self.essential_locations.items():
				is_residential = "/residential" in designation
				if is_residential:
					designation = designation[:-1*len("/residential")]
				if designation == 'residential':
					designation = ''
					is_residential = True
				out.write(f"{loc},{designation},{1 if is_residential else 0}\n")

		print("Created", output_file)
		return output_file

	def add_residences_to_location_designations(self, ) -> Dict[str, str]:
		essential_locations = self._assign_essential_locations()
		assigned_locations = dict()
		for pid, lid in essential_locations.items():
			assigned_locations[lid.pop()] = self.essential_workers[pid]
		for home in self.home_locations:
			if home in assigned_locations and not self.home_locations[home]:
				print(f"WARNING: Home location {home} already assigned as {assigned_locations[home]}")
				assigned_locations[home] = f'{assigned_locations[home]}/residential'
			else:
				assigned_locations[home] = 'residential'
		return assigned_locations

	def count_location_visitors(self) -> (Dict[str, int], Dict[str, int]):
		locations = defaultdict(list)
		for f in self.synth_pop_files['activities']:
			with open(f, 'r') as locations_in:
				headers = locations_in.readline()[:-1].split(",")
				lid_index = headers.index('lid')
				pid_index = headers.index('pid')
				for line in locations_in:
					activity = line[:-1].split(",")
					locations[activity[lid_index]].append(activity[pid_index])

		visits_per_location = dict()
		unique_visits_per_location = defaultdict(int)

		for loc, visits in locations.items():
			visits_per_location[loc] = len(visits)
			unique_visits_per_location[loc] = len(set(visits))

		return visits_per_location, unique_visits_per_location

	def mark_government_designation_with_highest_num_unique_visits_as_dmv(self):
		visits, unique = self.count_location_visitors()
		for lid in sorted(self.essential_locations.keys(), key=lambda x: unique[x], reverse=True):
			if self.essential_locations[lid] == 'government':
				self.essential_locations[lid] = 'dmv'
				print(f"Marked {lid} with {unique[lid]} unique and {visits[lid]} overall visits as DMV location")
				return

		print("WARNING: No government locations found: No DMV location set")

	def __load_essentials_from_activities(self):
		"""
		The activity files contain a column "location_designation". These designations are consistent
		across location ID's, but not across person ID's, suggesting these are the designations of locations.

		Most values in this column are of the form <string>:<string>, but in that case, the <string> before the
		colon is always 'none'. Presumably, this would be the designation of the agent before the colon and of the
		location after the colon.

		We extract the designation of the locations, by only taking the designation after the colon if a colon is present
		and otherwise assigning the string as the designation of that location.
		"""
		all_locations = set()
		locations = dict()
		for f in self.synth_pop_files['activities']:
			with open(f, 'r') as activities_in:
				headers = activities_in.readline()[:-1].split(",")
				lid = headers.index('lid')
				designation = headers.index('location_designation')

				for line in activities_in:
					activity = line[:-1].split(",")
					all_locations.add(activity[lid])
					des = activity[designation]
					if ":" in des:
						des = des[des.index(":") + 1:]
					if des != "" and des != 'none':
						locations[activity[lid]] = des

		return locations

	def find_fraction_of_essential_visits_by_activity_type(self):
		activity_types = ["TRIP", "HOME", "WORK", "SHOP", "OTHER", "SCHOOL", "COLLEGE", "RELIGIOUS"]

		visitors = defaultdict(lambda: defaultdict(lambda: {'essential': 0, 'nonessential': 0, 'dmv': 0, 'residential': 0}))
		for f in self.synth_pop_files['activities']:
			with open(f, 'r') as activities_in:
				headers = activities_in.readline()[:-1].split(",")
				lid = headers.index('lid')
				acttype = headers.index('activity_type')
				time = headers.index('start_time')
				for line in activities_in:
					activity = line[:-1].split(",")
					day = self.get_day_of_week_from_time(activity[time])
					# TODO ignore DMV locations probably
					des = self.essential_locations[activity[lid]] if activity[lid] in self.essential_locations else 'nonessential'
					if des not in ['nonessential', 'dmv', 'residential']:
						des = 'essential'
					visitors[day][activity[acttype]][des] += 1

		for day, visits in visitors.items():
			print(day)
			for type, items in visits.items():
				print('\t', activity_types[int(type)], "Essential:", items['essential'], "non-essential", items['nonessential'], 'DMV', items['dmv'], 'residential', items['residential'])
			items = visits[str(activity_types.index('OTHER'))]
			cancelling = items['nonessential'] + items['dmv']
			total = cancelling + items['essential'] + items['residential']
			print()
			print('\t', cancelling/total*100 if total > 0 else 0, "% of OTHER activities will be cancelled")

	@staticmethod
	def get_day_of_week_from_time(time: str or int):
		"""Returns day of the week in range 0-6, with 0 being monday"""
		return int(int(time) / (60 * 60 * 24))

	def find_intersection_between_calculated_and_listed_location_designations(self):
		listed_location_assignments = self.__load_essentials_from_activities()

		residentials = [x for x in self.essential_locations if self.essential_locations[x] == 'residential']
		non_residentials = [x for x in self.essential_locations if x not in residentials]

		essentials_not_in_pop = [x for x in non_residentials if x not in listed_location_assignments]
		pop_not_in_essentials = [x for x in listed_location_assignments if x not in non_residentials]

		pop_also_as_residential = [x for x in listed_location_assignments if x in residentials]

		print(f"{len(essentials_not_in_pop)} of {len(non_residentials)} were NOT in activity schedules")
		print(f"{len(pop_not_in_essentials)} of {len(listed_location_assignments)} listed were not in essential designation")
		print(f"{len(pop_also_as_residential)} of {len(listed_location_assignments)} were in pop, but are also residential")

		for x in non_residentials:
			if x in listed_location_assignments and self.essential_locations[x] != listed_location_assignments[x]:
				print(x, self.essential_locations[x], listed_location_assignments[x] if x in listed_location_assignments else '')

		assignments = defaultdict(int)
		for x in pop_not_in_essentials:
			assignments[listed_location_assignments[x]] += 1

		for x in assignments:
			print(x, assignments[x])

	def create_distribution_of_designation_bar_chart(self):
		listed_location_assignments = self.__load_essentials_from_activities()
		residentials = [x for x in self.essential_locations if self.essential_locations[x] == 'residential']
		non_residentials = [x for x in self.essential_locations if x not in residentials]

		set_difference_assignments = [x for x in listed_location_assignments if x not in non_residentials]

		worker_location_distribution = defaultdict(int)
		located_location_distribution = defaultdict(int)
		set_difference_distribution = defaultdict(int)

		for x in non_residentials:
			des = self.essential_locations[x]
			if "/residential" in des:
				print(des)
				des = des[:-1 * len("/residential")]
			worker_location_distribution[des] += 1

		for designation in listed_location_assignments.values():
			located_location_distribution[designation] += 1

		for x in set_difference_assignments:
			set_difference_distribution[listed_location_assignments[x]] += 1

		labels = list(set(list(worker_location_distribution.keys()) + list(located_location_distribution.keys())))
		worker = [worker_location_distribution[x] if x in worker_location_distribution else 0 for x in labels]
		located = [located_location_distribution[x] if x in located_location_distribution else 0 for x in labels]
		set_d = [set_difference_distribution[x] if x in set_difference_distribution else 0 for x in labels]

		worker_pct = [x / len(non_residentials) * 100 for x in worker]
		located_pct = [x / len(listed_location_assignments) * 100 for x in located]
		set_d_pct = [x / len(set_difference_assignments) * 100 for x in set_d]

		self.__plot_ranges(labels, worker, located, set_d, False)
		self.__plot_ranges(labels, worker_pct, located_pct, set_d_pct, True)

	@staticmethod
	def __plot_ranges(labels, r1, r2, r3, pct):
		x = np.arange(len(labels))
		width = 0.2

		fig, ax = plt.subplots()
		rects1 = ax.bar(x - width, r1, width, label="Essential Workers")
		rects2 = ax.bar(x, r2, width, label="Located Activities")
		rects3 = ax.bar(x + width, r3, width, label="Located Activities (without overlap)")

		ax.set_ylabel("Method of assigning designation")
		ax.set_xlabel(("Percentage" if pct else "Number") + " of locations with designation")
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend()

		ax.bar_label(rects1, padding=3)
		ax.bar_label(rects2, padding=3)
		ax.bar_label(rects3, padding=3)

		fig.tight_layout()
		title = "Distribution of location assignments"
		if pct:
			title += " (percentage)"
		plt.title(title)
		plt.show()

	def plot_location_visitors(self):
		visits, unique = self.count_location_visitors()

		per_num_visits = defaultdict(lambda: defaultdict(int))

		for loc, num in unique.items():
			per_num_visits[num]['total'] += 1
			designation = self.essential_locations[loc] if loc in self.essential_locations else 'none'
			if '/residential' in designation:
				designation = designation[: -1*len('/residential')]
			per_num_visits[num][designation] += 1

		keys = list(set([item for sublist in [per_num_visits[y].keys() for y in per_num_visits.keys()] for item in sublist]))
		keys.remove('total')
		keys.remove('none')
		x = np.arange(0, 250)

		values = [[per_num_visits[n][label] if n in per_num_visits and label in per_num_visits[n] else 0 for n in x] for label in keys]

		fig, ax = plt.subplots()
		for val in range(len(values)):
			print("Adding bar for " + keys[val])
			ax.bar(x, values[val], 1, label=keys[val])

		plt.legend()
		plt.yscale('log')
		plt.show()

if __name__ == "__main__":
	EDE = EssentialDesignationExtractor()
	EDE.from_synthetic_population_directory(sys.argv[1])

	# If you want to do some stats
	# EDE.find_fraction_of_essential_visits_by_activity_type()

	# If you want to figure out what the difference between essential workers and location designations is 
	# EDE.find_intersection_between_calculated_and_listed_location_designations()
	# EDE.create_distribution_of_designation_bar_chart()
	# EDE.plot_location_visitors()
