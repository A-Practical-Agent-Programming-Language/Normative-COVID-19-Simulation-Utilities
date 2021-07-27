import os
import random
import time
from typing import List, Dict


class StateFile(object):

    __state_file: str

    def __init__(
        self, counties, frac_infected: float = 0, frac_symptomatic: float = 0.6
    ):
        self.counties = counties
        self.frac_infected = frac_infected
        self.frac_symptomatic = frac_symptomatic

    def generate_from_fractions(self) -> str:
        """
        Regardless of whether a state file is already passed, generate a new state file for the current set of
        counties, using the fractions specified in the constructor
        Returns: Generated state file location
        """
        name = "start_state_counties_{0}_ifrac={1}_sfrac={2}.csv".format(
            self.__get_fips_list_string(), self.frac_infected, self.frac_symptomatic
        )
        self.__state_file = os.path.join(".persistent", name)

        if not os.path.exists(self.__state_file):
            os.makedirs(".persistent", exist_ok=True)
            for config in self.counties.values():
                self.__create_state_file(
                    *self.__infect_random(self.__read_persons(config))
                )
        return self.__state_file

    def merge_from_config(self) -> str:
        """
        Merge existing state files if present.
        For counties for which no state file is specified, a new state file will be generated with the fractions
        passed to the constructor
        Returns: generated state file location
        """
        counties = dict()
        for c in self.counties:
            if "statefile" in self.counties[c]:
                counties[c] = self.__load_state_file(self.counties[c]["statefile"])
            else:
                healthy, symptomatic, asymptomatic = self.__infect_random(
                    self.__read_persons(self.counties[c])
                )
                counties[c] = dict(
                    healthy=healthy,
                    symptomatic=symptomatic,
                    asymptomatic=asymptomatic,
                    frac_infected=self.frac_infected,
                    frac_symptomatic=self.frac_symptomatic,
                )

        healthy, symptomatic, asymptomatic = list(), list(), list()
        for c in counties:
            healthy += counties[c]["healthy"]
            symptomatic += counties[c]["healthy"]
            asymptomatic += counties[c]["asymptomatic"]

        self.__state_file = (
            "start_state_counties_"
            + self.__get_fips_list_string()
            + "_"
            + str(time.time())
            + ".csv"
        )
        self.__create_state_file(healthy, symptomatic, asymptomatic)

        return self.__state_file

    def __get_fips_list_string(self):
        return "_".join([str(self.counties[x]["fipscode"]) for x in self.counties])

    @staticmethod
    def __load_state_file(state_file_location: str) -> Dict[str, List[str] or float]:
        healthy, symptomatic, asymptomatic = list()
        with open(state_file_location, "r") as state_file_in:
            for line in state_file_in:
                pid, _, state = line[:-1].split(",")
                if state == "2":
                    symptomatic.append(pid)
                elif state == "1":
                    asymptomatic.append(pid)
                else:
                    healthy.append(pid)

        return dict(
            healthy=healthy,
            symptomatic=symptomatic,
            asymptomatic=asymptomatic,
            frac_infected=(len(symptomatic) + len(asymptomatic))
            / (len(healthy) + len(symptomatic) + len(asymptomatic)),
            frac_symptomatic=len(symptomatic) / (len(symptomatic) + len(asymptomatic)),
        )

    @staticmethod
    def __read_persons(config):
        persons = list()
        for person_file in config["persons"]:
            with open(person_file, "r") as persons_in:
                headers = persons_in.readline()[:-1].split(",")
                lid_index = headers.index("pid")
                for line in persons_in:
                    persons.append(line[:-1].split(",")[lid_index])

        return persons

    def __infect_random(self, persons):
        num_infected = len(persons) * self.frac_infected
        num_symptomatic = int(num_infected * self.frac_symptomatic)
        num_infected = int(num_infected)

        infected = random.sample(persons, num_infected)
        symptomatic = random.sample(infected, num_symptomatic)
        asymptomatic = [x for x in infected if x not in symptomatic]
        symptomatic = [x for x in symptomatic if x not in asymptomatic]
        healthy = [x for x in persons if x not in infected]
        assert set(healthy).union(set(asymptomatic).union(set(symptomatic))) == set(
            persons
        )
        assert len(healthy) + len(asymptomatic) + len(symptomatic) == len(persons)
        assert set(asymptomatic).union(set(symptomatic)) == set(infected)
        assert len(asymptomatic) + len(symptomatic) == len(infected)

        return healthy, symptomatic, asymptomatic

    def __create_state_file(
        self, healthy: List[str], symptomatic: List[str], asymptomatic: List[str]
    ):
        with open(self.__state_file, "w") as state_out:
            for person in healthy:
                state_out.write("{0},0,0\n".format(person))
            for person in symptomatic:
                state_out.write("{0},0,2\n".format(person))
            for person in asymptomatic:
                state_out.write("{0},0,3\n".format(person))
