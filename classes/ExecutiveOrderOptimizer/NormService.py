from datetime import datetime, timedelta
import math
from collections import defaultdict
from typing import List, Tuple

DATE_FORMAT = "%Y-%m-%d"


class Norm(object):

    def __init__(self, name: str, start: int, duration: int, index: int, params: str or None, paper_params: str or None):
        self.name = name
        self.index = index
        self.params = params
        self.paper_params = paper_params
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
        return f"{self.name}{params} ({self.start_date} - {self.end_date})"

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__()


norms = {
    "EO0":
        [
            (1, "AllowWearMask", None, None),
            (4, "EncourageTelework", None, None)
        ],
    "EO1":
        [
            (7, "SchoolsClosed", "K12", "K12")
        ],
    "EO2":
        [
            (8, "SmallGroups", "100,public", "100, \mathit{public}")
        ],
    "EO3":
        [
            (6, "ReduceBusinessCapacity", "10", "10"),
            (2, "BusinessClosed", "7 DMV offices", "DMV"),
            (8, "SmallGroups", "10,public", "10, \mathit{public}"),
            (9, "StayHome", "age>65", "\mathit{age}\geq 65"),
            (9, "StayHomeSick", None, None),
            (5, "EncourageSocialDistance", None, None)
        ],
    "EO4":
        [
            (2, "BusinessClosed", "7 DMV offices;NEB", "\mathit{DMV\wedge NEB}"),
            (9, "StayHome", "all", "\mathit{all}"),
            (10, "TakeawayOnly", None, None)
        ],
    "EO5":
        [
            (5, "MaintainDistance", None, None),
            (7, "SchoolsClosed", "K12;HIGHER_EDUCATION", "\mathit{K12\ or\ HE}"),
            (8, "SmallGroups", "10,PP", "10, \mathit{all}")
        ],
    "EO6":
        [
            (3, "EmployeesWearMask", None, None),
            (6, "ReduceBusinessCapacity", "50%", "50\%")
        ],
    "EO7":
        [
            (11, "WearMasInPublicIndoor", None, None)
        ],
    "EO8":
        [
            (7, "SchoolsClosed", "K12", "\mathit{K12}"),
            (8, "SmallGroups", "50,PP", "50, \mathit{all}")
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
    start += timedelta(weeks=data_point)
    return start.strftime(DATE_FORMAT)


def data_point_from_date(to_convert: datetime) -> float:
    return float(days_between_dates(datetime(year=2020, month=3, day=1), to_convert)) / 7


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
            new_norm = Norm(norm.name, norm.start, norm.end - norm.start, norm.index, param, None)
            norm_list.append(new_norm)
    else:
        norm_list.append(norm)
    return norm_list


def test_data_point_from_date():
    start = datetime(year=2020, month=3, day=1)
    not_converting_correctly = 0
    while start < datetime(year=2020, month=6, day=30):
        converted = date_from_data_point(data_point_from_date(start))
        if converted != start.strftime(DATE_FORMAT):
            print(start.strftime(DATE_FORMAT), converted, converted == start.strftime(DATE_FORMAT))
            not_converting_correctly += 1
        start = start + timedelta(days=1)

    print(f"Done. {not_converting_correctly} items did not convert correctly. If this value is 0, data_point_from_time works correctly!")

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
    # print(get_bounds())
    # find_duplicate_norms_in_eos()
    test_data_point_from_date()
