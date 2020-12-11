import sys
import math


class Epicurve(object):

	epicurve_headers = ["succ", "expo", "isymp", "iasymp", "recov"]

	def __init__(self, epicurve_file, target_r=1.5):
		self.epicurve_file = epicurve_file
		self.epicurve = self.read_file()
		self.target_r = target_r

	def read_file(self):
		"""
		Reads an epicurve file as outputted by PanSim
		"""
		results = list()
		with open(self.epicurve_file, 'r') as f:
			for line in f.read().splitlines()[1:]:
				split_char = ";" if ";" in line else ","
				results.append(dict(zip(self.epicurve_headers, map(lambda x: int(x), line.split(split_char)))))

		return results

	@staticmethod
	def get_infected(epicurve_day):
		return epicurve_day["isymp"] + epicurve_day["iasymp"]

	def get_population_size(self):
		return sum(map(lambda x: self.epicurve[0][x], self.epicurve[0].keys()))

	# ScoreMethod
	def r_0(self):
		"""Simplified method to calculate the relative increase in infections from day to day,
		which I think is an approximation for the r-zero value.

		Rationale: I think this R-value should be higher than 1 (at least at the start of the simulation)
		but not be too high (e.g. max 3 in the extreme case)

		Returns a list with this value for each day
		"""
		r_0 = list()
		last_total = 0
		last_delta = self.get_infected(self.epicurve[0])
		for day in self.epicurve:
			infected = self.get_infected(day)
			delta = infected - last_total
			r_0.append(delta / last_delta if last_delta > 0 else 0)
			last_total = infected
			last_delta = delta

		return r_0

	# ScoreMethod
	def average_r_zero(self):
		"""
		Calculate the average of all relevant r0 values. Returns a tuple, where the
			first element is the fraction of calculated r0 values that were relevant,
			and the second tuple is the average of those relevant values.

			An r-0 value is considered relevant if it is non-zero, and the infection has
			not died out yet at that point due to herd immunity
		Returns:
			...
		"""
		until = self.days_before_herd_immunity()
		r_0 = self.r_0()
		r_0_relevant = list(filter(lambda x: x != 0, r_0[1:until]))
		return len(r_0_relevant) / len(r_0), sum(r_0_relevant) / len(r_0_relevant) if len(r_0_relevant) > 0 else 0

	# ScoreMethod
	def get_pcts_infected(self):
		"""
		Find the fraction of the total population that is infected for each day in the epicurve

		Rationale: Samarth said this value should probably not exceed 1% for any given day

		Returns a list of fractions of the same size as the epicurve
		"""
		infected = list()
		population_size = self.get_population_size()
		for day in self.epicurve:
			i = self.get_infected(day)
			infected.append(i / population_size)
		return infected

	# ScoreMethod
	def days_before_herd_immunity(self):
		"""
		Finds the index of the first day in the epicurve where there are no infections at all.

		Rationale being that this number being low(er than 120 days we simulate) indicates too aggressive disease progression.

		Returns index of epicurve, so higher is better.
		"""
		for i, day in enumerate(self.epicurve):
			if self.get_infected(day) + day["expo"] == 0:
				return i

		# If none found, award maximum score
		return len(self.epicurve)

	# coreMethod
	@staticmethod
	def score_average_r(average_r, target_r):
		"""Highest score of 1 when average_r is target _r, equal penalty for higher or lower average r"""
		x = average_r - target_r
		return -1 * x * x + 1

	# ScoreMethod
	@staticmethod
	def score_infected(pct_infected):
		"""
		Asign two scores to the fraction of the population being infected during the progression of the model.

		Methods used are plotted at https://www.desmos.com/calculator/qjsefdiujd

		Returns:
			stddev_score: Lower standard deviation in number of infections between days (indicating stable progression)
							is assigned a higher score. Score is normalized between 0 and 1. Higher is better
			max_infected_score: Assigns a score the the largest found fraction of infected people on any given day. This
								means this score selects only one day from the epicurve to consider. It assigns a higher
								value if the score is roughly 0.02 (i.e. roughly 2% maximum infected during the entire
								simulation) and quickly decreases when the fraction becomes higher than that.
								Higher is better.
		"""
		stdev = math.sqrt(sum(map(lambda x: x*x, pct_infected)))

		# Normalize between 0 and 1
		stddev_score = 2 + -2 * (1 / (1 + math.pow(math.e, -1 * stdev)))

		# Award maximum of 2% of population infected at one time
		max_infected_score = 1 - math.pow(3 * (max(pct_infected) - 0.06), 2)

		return stddev_score, max_infected_score

	def get_score(self):
		"""
		Read an epicurve file and assign a single score value. Target_r can be used to fit the
		approximation of the r-zero value found in the epicurve
		"""
		# relevant_fraction_r, average_r = average_r_zero(epicurve)
		days_before_immune = self.days_before_herd_immunity()
		# immune_score = 1 - (days_before_immune / len(self.epicurve))
		infection_num_variation, max_infected_score = self.score_infected(self.get_pcts_infected())
		# average_r_score = score_average_r(average_r, target_r)
		# return immune_score + average_r_score + infection_num_variation + max_infected_score
		return 150 - days_before_immune - max_infected_score

if __name__ == "__main__":
	epicurve = Epicurve(sys.argv[1])
	r0avg_relevant, r0avg = epicurve.average_r_zero()
	pct_infected = epicurve.get_pcts_infected()

	print("{0} agents in population, {1} days in the epicurve".format(epicurve.get_population_size(), len(epicurve.epicurve)))
	print("R-zero is {0} on average. Targetting an average r-zero of {1} this yields a score of {3} ({2} of all r-0 values were relevant)"
		.format(r0avg, 1.5, r0avg_relevant, Epicurve.score_average_r(r0avg, 1.5)))
	print("No more infected agents after {0} days".format(epicurve.days_before_herd_immunity()))
	print("Infection score is {0} for the standard deviation and {1} for the maximum number of infected agents"
		.format(*Epicurve.score_infected(pct_infected)))

	print("Loss value for this epicurve file is", epicurve.get_score())

	print("")
	print("R-zero values:", epicurve.r_0())
	print("")
	print("Fraction infected:", pct_infected)
