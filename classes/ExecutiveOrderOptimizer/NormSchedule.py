import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .NormService import norms, Norm, days_between_strings, split_param_groups

NormByDate = Dict[str, Dict[str, List[Norm]]]
NormEvents = Dict[str, List[Norm]]

# This class converts the output of the Bayesian optimization (i.e., a vector of start and end dates for each
# EO) to a norm schedule, and can write that schedule to a file that can be read by the simulation framework


class NormSchedule(object):

    def __init__(self, x: Dict[str, int], last_simulation_date: str or None = None, ):
        self.last_simulation_date = last_simulation_date
        self.grouped_norms = dict()
        for norm_name, norm_list in self.bayesian_output_to_grouped_norms(x).items():
            self.grouped_norms[norm_name] = self._resolve_norm_list(norm_list)
        self.norm_events = self.__create_event_list()

    @staticmethod
    def bayesian_output_to_grouped_norms(x: Dict[str, int]) -> Dict[str, List[Norm]]:
        grouped_norms = defaultdict(list)
        for i in range(9):
            start, duration = x[f'EO{i}_start'], x[f'EO{i}_duration']
            for norm in norms[f'EO{i}']:
                norm_obj = Norm(norm[1], start, duration, norm[0], norm[2])
                grouped_norms[norm[1]].append(norm_obj)

        return grouped_norms

    @staticmethod
    def _resolve_norm_list(norm_list: List[Norm]) -> List[Norm]:
        """
        Recursively resolves a list of norms, by updating start and end dates of norms so that they
        no longer overlap
        Args:
            norm_list: List of norms to resolve (most useful if they are all of the same type)

        Returns:
            List of norms without overlap
        """
        updated_norm_list: List[Norm] = list()
        overlap = NormSchedule.__find_overlap(norm_list)
        for overlapping in overlap:
            if len(overlapping) == 1:
                updated_norm_list += overlapping
            else:
                norm1 = overlapping[0]
                norm2 = overlapping[1]
                if norm1.params == norm2.params:
                    overlapping.pop(0)
                    norm2.start = min(norm1.start, norm2.start)
                    norm2.end = max(norm1.end, norm2.end)
                    overlapping[0] = norm2
                else:
                    if norm1.end > norm2.end:
                        norm3 = Norm(norm1.name, norm2.end, norm1.end - norm2.end, norm1.index, norm1.params)
                        overlapping.append(norm3)
                    norm1.end = norm2.start
                    if norm1.start == norm1.end:
                        overlapping.pop(0)
                    else:
                        overlapping[0] = norm1
                updated_norm_list += NormSchedule._resolve_norm_list(overlapping)

        return updated_norm_list

    @staticmethod
    def __find_overlap(norm_list: List['Norm']) -> List[List['Norm']]:
        """
        In a list of norms, finds all norms that overlap (i.e., one norm starts before the other ends, and vice
        versa

        Args:
            norm_list: List of norms to find overlap for (most useful if they are all of the same type)

        Returns:
            List of lists, with each sublist containing norms that have some form of overlap
            Note that not all norms have to overlap pairwise, because norm a and c can both overlap with
            a norm b, but not with each other
        """
        norm_list = sorted(norm_list, key=lambda x: (x.start, x.end))
        overlap = list()
        i = 0
        while i < len(norm_list):
            start, end = norm_list[i].start, norm_list[i].end
            aggregator = [norm_list[i]]
            i += 1
            for j in range(i, len(norm_list)):
                if norm_list[i].overlaps(start, end, norm_list[j]):
                    aggregator.append(norm_list[j])
                    end = max(end, norm_list[j].end)
                    i += 1
                else:
                    break
            overlap.append(aggregator)
        return overlap

    def get_active_duration(self, norm_string: str):
        active_duration = 0
        if norm_string in self.grouped_norms:
            for norm in self.grouped_norms[norm_string]:
                active_duration += days_between_strings(norm.start_date, norm.end_date, self.last_simulation_date)

        return active_duration

    def __create_event_list(self):
        event_list = defaultdict(lambda: dict(start=[], end=[]))
        for norm_list in self.grouped_norms.values():
            for norm in norm_list:
                event_list[norm.start]['start'].append(norm)
                event_list[norm.end]['end'].append(norm)

        return event_list

    def write_to_file(self, filename: str):
        os.makedirs(Path(filename).parent.absolute(), exist_ok=True)
        with open(filename, 'w') as fout:
            fout.write('start,norm,end,param,statement\n')
            for norm_event in sorted(list(self.norm_events.keys())):
                for norm_instance in self.norm_events[norm_event]['start']:
                    for norm in split_param_groups(norm_instance):
                        fout.write(f"{norm.start_date},{norm.name},{norm.end_date},")
                        if norm.params is not None:
                            fout.write(f'"{norm.params}"')
                        fout.write(",\n")


def test_norm_resolution():
    lst: List[Norm] = [
        Norm('test', 1, 11, 1, ['params1']),
        Norm('test', 2, 3, 1, ['params2']),
        Norm('test', 3, 8, 1, ['params3']),
        Norm('test', 11, 13, 1, ['params4']),
    ]
    print(NormSchedule._resolve_norm_list(lst))


def test_norm_schedule():
    x_start = 17
    x_end = 17
    args = dict()
    for EO_index in range(9):
        args[f'EO{EO_index}_start'] = x_start
        args[f'EO{EO_index}_duration'] = x_end
        # x_start += 1
        # x_end += 1
    ns = NormSchedule(args)
    ns.write_to_file('test.csv')


if __name__ == "__main__":
    # test_norm_resolution()
    test_norm_schedule()