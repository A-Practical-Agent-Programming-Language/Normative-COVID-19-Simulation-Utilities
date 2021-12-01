import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from .NormService import norms, Norm, days_between_strings, split_param_groups, DATE_FORMAT, data_point_from_date, \
    date_from_data_point

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
    def from_norm_schedule(norm_schedule_file: str, last_simulation_date: str or None = None, until=None):
        """In order to score a run for which the norm schedule was not the result of a policy,
        we need to create a NormSchedule object from the norm_schedule csv file.

        This is not trivial without breaking existing code. So this is the way we do it
        """
        base_norms = dict()
        for i in range(9):
            base_norms[f"EO{i}_start"] = 0
            base_norms[f"EO{i}_duration"] = 1
        ns = NormSchedule(base_norms, last_simulation_date)
        ns.grouped_norms = defaultdict(list)
        ns.norm_events = defaultdict(list)

        with open(norm_schedule_file, 'r') as norms_in:
            for line in norms_in:
                m = re.findall(r'(2020-\d{2}-\d{2}),(\w+),(2020-\d{2}-\d{2})?,("?)([\w\d>, %]*?)?\4,("?).*\6,("?).*\7,', line)
                if len(m):
                    m = m[0]
                    start, name, end, _, params, _, _ = m
                    if until is None or start <= until:
                        if params == '':
                            params = None
                        elif "," in params:
                            params = params.replace(" ", "")
                        if end == '':
                            end = (datetime.strptime(last_simulation_date, DATE_FORMAT) + timedelta(days=10)).strftime(DATE_FORMAT)
                        duration = float(days_between_strings(start, end)) / 7
                        start = data_point_from_date(datetime.strptime(start, DATE_FORMAT))
                        norm = Norm(name, start, duration, 0, params, params)
                        ns.grouped_norms[name].append(norm)


        # TODO resolve:
        #  - BusinessClosed (DMV + NEB in same)
        #  - SchoolsClosed (K12 + HIGHER_EDUCATION)
        to_resolve = ["BusinessClosed", "SchoolsClosed"]
        for resolving in to_resolve:
            new_list = list()
            norms = sorted(ns.grouped_norms[resolving], key=lambda x: (x.start, x.end))
            for i in range(len(norms)):
                for j in range(i+1, len(norms)):
                    first_norm = ns.grouped_norms[resolving][i]
                    second_norm = ns.grouped_norms[resolving][j]
                    if first_norm.end > second_norm.start:
                        previous_end = first_norm.end
                        first_norm.end = second_norm.start
                        second_norm.params = first_norm.params + ";" + second_norm.params
                        if date_from_data_point(previous_end) > date_from_data_point(second_norm.end):
                            new_list.append(Norm(first_norm.name, second_norm.end, previous_end - second_norm.end, first_norm.index, first_norm.params, first_norm.paper_params))
                    new_list.append(first_norm)
            if len(ns.grouped_norms[resolving]):
                new_list.append(ns.grouped_norms[resolving][-1])

            # print(sorted(new_list, key=lambda x: (x.start, x.end)))
            # print("\n\n")

            ns.grouped_norms[resolving] = new_list

        for norm_name, norms in ns.grouped_norms.items():
            for norm in norms:
                ns.norm_events[norm.start].append(norm)

        return ns

    @staticmethod
    def bayesian_output_to_grouped_norms(x: Dict[str, int]) -> Dict[str, List[Norm]]:
        grouped_norms = defaultdict(list)
        for i in range(9):
            start, duration = x[f'EO{i}_start'], x[f'EO{i}_duration']
            for norm in norms[f'EO{i}']:
                norm_obj = Norm(norm[1], start, duration, norm[0], norm[2], norm[3])
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
                        norm3 = Norm(norm1.name, norm2.end, norm1.end - norm2.end, norm1.index, norm1.params, norm1.paper_params)
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

    @staticmethod
    def norm_ident(norm: Norm):
        return norm.name + (norm.params if norm.params is not None else '')

    @staticmethod
    def data_point_to_tikz_x_coordinate(data_point: int, min_data_point: int, max_data_point: int, max_x_coordinate: int):
        return (data_point - min_data_point) * (max_x_coordinate / (max_data_point - min_data_point))

    def to_tikz(self, output_file=None):
        line = 0
        node = 0

        min_data_point = min(self.norm_events.keys())
        max_data_point = 17
        max_tickz_x = 15

        last_line_for_event = dict()
        if output_file is None:
            outf = sys.stdout
        else:
            outf = open(output_file, 'w')
        outf.write("\\begin{figure}\n\\resizebox{\linewidth}{!}{%\n\\begin{tikzpicture}\n")
        outf.write("\\tikzstyle{node_style}=[fill=black, draw=black, shape=circle, scale=.3]\n")
        for i, (eo, norm_list) in enumerate(norms.items()):
            outf.write(f"% EO{i+1}")
            found_norms = defaultdict(lambda: defaultdict(list))
            for norm in norm_list:
                for instantiated_norm in self.grouped_norms[norm[1]]:
                    if norm[2] == instantiated_norm.params and instantiated_norm.start < max_data_point:
                        found_norms[instantiated_norm.start][instantiated_norm.end].append(instantiated_norm)

            norm_on_line = dict()
            skip_norm_line = list()
            written_EO = False
            for j, start in enumerate(found_norms):
                # print("EO", i, start, found_norms[start])
                if start > max_data_point:
                    print(found_norms[start])
                    continue
                for end, found_norms_for_this_end in found_norms[start].items():
                    found_norm = found_norms_for_this_end[0]
                    norm_end = min(max_data_point, found_norm.end)

                    # pos_start = (17 - min_data_point) / 10 * (found_norm.start - min_data_point)
                    # pos_end = (17-min_data_point)/10*(norm_end-min_data_point)
                    pos_start = self.data_point_to_tikz_x_coordinate(found_norm.start, min_data_point, max_data_point, max_tickz_x)
                    pos_end = self.data_point_to_tikz_x_coordinate(norm_end, min_data_point, max_data_point, max_tickz_x)


                    outf.write("% {0}-{1}  ({2}  -  {3})\n".format(
                        found_norm.start, found_norm.end,
                        date_from_data_point(found_norm.start),
                        date_from_data_point(found_norm.end)
                    ))
                    color = 'black'  # if found_norm.end < max_data_point else 'green' if found_norm.start < max_data_point else 'red'

                    this_line = line

                    if self.norm_ident(found_norm) in norm_on_line:
                        this_line = norm_on_line[self.norm_ident(found_norm)]
                        skip_norm_line.append(self.norm_ident(found_norm))
                        print("Reusing line", this_line, "for", self.norm_ident(found_norm))
                    else:
                        norm_on_line[self.norm_ident(found_norm)] = this_line

                    outf.write(f"\\node [style=node_style] ({node}) at ({pos_start}, {-1 * this_line * .3:.2}) {{}};\n")
                    node += 1
                    style = "{draw=none}" if end > max_data_point else 'node_style'
                    outf.write(f"\\node [style={style}] ({node}) at ({pos_end}, {-1 * this_line * .3:.2}) {{}};\n")
                    node += 1
                    line += 1
                    outf.write(f"\\draw[{color}] ({node - 2}) to ({node - 1});")
                    last_line_for_event[pos_start] = (node-2, found_norm.start)
                    last_line_for_event[pos_end] = (node-1, found_norm.end)
                    norm_name_list = list()
                    print(f"EO{i+1}  {date_from_data_point(start)}  -- {date_from_data_point(end)}  ({start}-{end}) plotted at ({pos_start},{pos_end}). Max data point is {max_data_point}")
                    for found_norm_obj in found_norms_for_this_end:
                        if self.norm_ident(found_norm_obj) not in skip_norm_line:
                            params = f"({found_norm_obj.paper_params})" if found_norm_obj.paper_params is not None else ''
                            norm_name_list.append(f"n_{{{found_norm_obj.index}}}{params}")

                    if len(norm_name_list):
                        EO_name = f"EO$_{{{i+1}}}:" if j == 0 else "\ \ \ \ \ \ \ $"
                        outf.write(f"\\node[draw=none,style={{anchor=west, inner sep=0pt}}] at (-3,{-1 * (line-1) * .3:.2}) {{{EO_name} {','.join(norm_name_list)}$}};\n")
                        written_EO = True
                    else:
                        print("Not printing EO", i, found_norms_for_this_end)
                        line -= 1
            if not written_EO:
                norm_name_list = list()
                for found_norm_obj in norms[f"EO{i}"]:
                    params = f"({found_norm_obj[3]})" if found_norm_obj[3] is not None else ''
                    norm_name_list.append(f"n_{{{found_norm_obj[0]}}}{params}")
                EO_name = f"EO$_{{{i + 1}}}:"
                print(EO_name, norm_name_list)
                outf.write(f"\n\\node[draw=none,style={{anchor=west, inner sep=0pt}},color=gray] at (-3,{-1 * line * .3:.2}) {{{EO_name} {','.join(norm_name_list)}$}};\n")
                line += 1
            outf.write("\n")

        print(last_line_for_event)
        for x_pos, (node, data_point) in last_line_for_event.items():
            if data_point <= max_data_point:
                color, date_str = self.convert_date(data_point)
                outf.write(f"\draw [dashed] ({node}) -- ({x_pos},0.3);\n")
                outf.write(f"\\node [draw=none,style={{anchor=east,inner sep=0pt}}] at ({x_pos}, 0.48) {{{date_str}}};\n")
            else:
                print("Skipping", x_pos, node, data_point, "because larger than", max_data_point, data_point - max_data_point)

        outf.write("\\end{tikzpicture}\n}\n\caption{Timeline of the best found policy}\n\label{fig:eos-after-optimization}\n\end{figure}")
        if output_file is not None:
            outf.close()

    months_seen = list()

    def convert_date(self, data_point):
        date = date_from_data_point(data_point)[5:]
        repl = {3: ('Mar', 'red'), 4: ('Apr', 'teal'), 5: ('May', 'orange'), 6: ('Jun', 'black')}
        month = int(date[0:2])
        day = int(date[3:])

        date_str = f"{repl[month][0]} {day}"
        if month in self.months_seen:
            date_str = str(day)
        else:
            self.months_seen.append(month)

        return repl[month][1], date_str



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
    # test_norm_schedule()
    # NormSchedule.from_norm_schedule("/home/jan/dev/university/2apl/simulation/sim2apl-episimpledemics/src/main/resources/norm-schedule.csv", "2020-06-28")
    ns = NormSchedule(
        # {  # DEFAULT_WEIGHTS
        #     'EO0_duration': 5, 'EO0_start': 16,
        #     'EO1_duration': 7, 'EO1_start': 6,
        #     'EO2_duration': 6, 'EO2_start': 8,
        #     'EO3_duration': 4, 'EO3_start': 11,
        #     'EO4_duration': 4, 'EO4_start': 17,
        #     'EO5_duration': 15, 'EO5_start': 11,
        #     'EO6_duration': 1, 'EO6_start': 12,
        #     'EO7_duration': 3, 'EO7_start': 13,
        #     'EO8_duration': 11, 'EO8_start': 8
        #  },
        { # after holiday fix: Default weights
            'EO0_duration': 5, 'EO0_start': 1, 'EO1_duration': 13, 'EO1_start': 9, 'EO2_duration': 5, 'EO2_start': 1,
            'EO3_duration': 3, 'EO3_start': 17, 'EO4_duration': 11, 'EO4_start': 14, 'EO5_duration': 7, 'EO5_start': 0,
            'EO6_duration': 5, 'EO6_start': 4, 'EO7_duration': 10, 'EO7_start': 14, 'EO8_duration': 11, 'EO8_start': 2
        },
        "2020-02-28"
    )
    ns.to_tikz("../../test-picture/test-picture.tex")

