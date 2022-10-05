"""
I did not think this through properly, but the print of the best score does not correspond to file or
directory names. This script is an easy way to convert between the two
"""
import re
from collections import defaultdict

score = """
Score 428.16063164842745
	 EO0_duration 2
	 EO0_start 9
	 EO1_duration 11
	 EO1_start 9
	 EO2_duration 9
	 EO2_start 2
	 EO3_duration 2
	 EO3_start 9
	 EO4_duration 5
	 EO4_start 8
	 EO5_duration 2
	 EO5_start 1
	 EO6_duration 6
	 EO6_start 7
	 EO7_duration 12
	 EO7_start 17
	 EO8_duration 11
	 EO8_start 2

"""

eo_names = list()
eo_vals = defaultdict(dict)
eo_dict = dict()

for line in score.splitlines():
    match = re.findall(r"\s*EO(\d+)_(\w+)\s+(\d+)", line)
    if len(match):
        eo_names.append(int(match[0][0]))
        eo_vals[int(match[0][0])][match[0][1]] = int(match[0][2])
        eo_dict[f"EO{match[0][0]}_{match[0][1]}"] = int(match[0][2])

eo_names = sorted(list(set(eo_names)))
eo_strings = "-".join(map(lambda i: "EO{0}_{1}_{2}".format(i, eo_vals[i]['start'], eo_vals[i]['duration']), eo_names))

print("norm-schedule-policy-{0}-run*.csv".format(eo_strings))
print("0.00042692107332804657l-0.6622033817409958c-0.03423684636507255f-110fs-run0-" + eo_strings)
print(eo_dict)