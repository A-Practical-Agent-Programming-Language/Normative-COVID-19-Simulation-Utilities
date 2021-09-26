from datetime import datetime, timedelta
import math
from collections import defaultdict
from typing import List, Tuple

DATE_FORMAT = "%Y-%m-%d"


class Norm(object):

    def __init__(self, name: str, start: int, duration: int, index: int, params: str or None):
        self.name = name
        self.index = index
        self.params = params
        self.start = start
        self.end = self.start + duration

    @property
    def start_date(self) -> str:
        return date_from_data_point(self.start)

    @property
    def end_date(self) -> str:
        return date_from_data_point(self.end)

    @staticmethod
    def overlaps(start, end, other: 'Norm'):
        return start < other.end and end > other.start

    def __str__(self):
        params = f" [{self.params}]" if self.params is not None else ''
        return f"{self.name}{params} ({self.start}-{self.end})"

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__()


norms = {
    "EO0":
        [
            (1, "AllowWearMask", None),
            (4, "EncourTelework", None)
        ],
    "EO1":
        [
            (7, "SchoolsClosed", "K12")
        ],
    "EO2":
        [
            (8, "SmallGroups", "100,public")
        ],
    "EO3":
        [
            (6, "RedBusinessCapac", "10"),
            (8, "SmallGroups", "10,public"),
            (9, "StayHome", "age>65"),
            (9, "StayHomeSick", None)
        ],
    "EO4":
        [
            (2, "BusinessClosed", "DMV;NEB"),
            (9, "StayHome", "all"),
            (10, "TakeawayOnly", None)
        ],
    "EO5":
        [
            (5, "MaintainDistance", None),
            (7, "SchoolsClosed", "K12;HIGHER_EDUCATION"),
            (8, "SmallGroups", "10,all")
        ],
    "EO6":
        [
            (3, "EmplWearMask", None),
            (6, "RedBusinessCapac", "50%")
        ],
    "EO7":
        [
            (11, "WearMaskPublInd", None)
        ],
    "EO8":
        [
            (7, "SchoolsClosed", "K12"),
            (8, "SmallGroups", "50,all")
        ]
}

def get_bounds():
    bounds = dict()
    for eo in norms:
        bounds[f"{eo}_start"] = (0, 17)
        bounds[f"{eo}_duration"] = (1, 17)  # Intentionally can be later than simulation end
    return bounds


def date_from_data_point(data_point: float):
    """
    Convert a data point as given by the Bayesian optimization to a start date. Each value that the data point
    corresponds to a week in the simulation time, i.e, 0 equals to 2020-03-01, 1 equals 2020-03-07, etc.

    The range is 0 - 16
    Args:
        data_point: Start week (provided by Bayesian optimization)

    Returns:
        Date string
    """
    start = datetime(year=2020, month=3, day=1)
    start += timedelta(weeks=int(data_point))
    return start.strftime(DATE_FORMAT)


def days_between_strings(start_date: str, end_date: str, max_date: None or str = None):
    return days_between_dates(
        datetime.strptime(start_date, DATE_FORMAT),
        datetime.strptime(end_date, DATE_FORMAT),
        datetime.strptime(max_date, DATE_FORMAT) if max_date is not None else None
    )


def days_between_dates(start_date: datetime, end_date: datetime, max_date: None or datetime = None):
    if max_date is not None:
        start_date = min(start_date, max_date)
        end_date = min(end_date, max_date)
    return math.fabs((start_date - end_date).days)  # We don't care which date is earlier


def split_param_groups(norm: 'Norm') -> List['Norm']:
    norm_list = list()
    if norm.params is not None and ";" in norm.params:
        for param in norm.params.split(";"):
            new_norm = Norm(norm.name, norm.start, norm.end, norm.index, param)
            norm_list.append(new_norm)
    else:
        norm_list.append(norm)
    return norm_list


def find_duplicate_norms_in_eos():
    norm_names = defaultdict(list)
    for EO, eo_norms in norms.items():
        for eo_norm in eo_norms:
            norm_names[eo_norm[1]].append((EO, eo_norm[2]))

    print("\nThe following norms occur more than once:")
    for norm, duplicate_list in norm_names.items():
        if len(duplicate_list) > 1:
            print("\t", norm, duplicate_list)


if __name__ == "__main__":
    print(get_bounds())
    find_duplicate_norms_in_eos()
