import os
import random
import string
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import click
import toml

from classes.EssentialLocations import EssentialDesignationExtractor
from classes.ExecutiveOrderOptimizer.covid_policy_bayes_opt.bayes_policy_opt import IntArray


def make_file_absolute(relative_to: str, file: str) -> (str, bool):
    """
    Make a passed file path absolute; relative to some other file path if not already absolute
    Args:
            relative_to: File location for reference
            file:        Path to file to make absolute

    Returns:
            1) Absolute file path, where the path is interpreted relative to {relative_to} if not already absolute
            2) Boolean indicating whether the absolute path exists
    """
    if os.path.isabs(file):
        return file, os.path.exists(file)
    else:
        resolved_path = os.path.abspath(os.path.join(relative_to, file))
        return resolved_path, os.path.exists(resolved_path)


def load_toml_configuration(county_config_file):
    """
    Parse the county configuration TOML file
    Args:
            county_config_file: Location of TOML config file

    Returns:
            Dictionary containing the parsed configuration
    """
    missing = list()
    conf = toml.load(county_config_file)
    county_config_location = os.path.dirname(county_config_file)
    norms, norms_exist = make_file_absolute(
        county_config_location, conf["simulation"]["norms"]
    )
    if not norms_exist:
        missing.append(("simulation.norms", [conf["simulation"]["norms"]]))
    conf["simulation"]["norms"] = norms

    disease, disease_exists = make_file_absolute(
        county_config_location, conf["simulation"]["diseasemodel"]
    )
    if not disease_exists:
        missing.append(("simulation.disease", [conf["simulation"]["diseasemodel"]]))
    conf["simulation"]["diseasemodel"] = disease

    for county, cconf in conf["counties"].items():
        for k in [
            "activities",
            "households",
            "persons",
            "locations",
            "statefile",
            "locationDesignations",
        ]:
            if k not in cconf:
                continue
            missing_in_key = []
            updated = []
            for f in cconf[k] if type(cconf[k]) is list else [cconf[k]]:
                path, exists = make_file_absolute(county_config_location, f)
                updated.append(path)
                if not exists:
                    missing_in_key.append(f)
            if len(missing_in_key):
                missing.append((f"counties.{county}.{k}", missing_in_key))
            conf["counties"][county][k] = updated

        conf["counties"][county] = ensure_location_designation_present(
            conf["counties"][county]
        )

    if len(missing):
        error = f"Some paths specified in {county_config_location} could not be resolved:\n\n"
        for (m, m_) in missing:
            error += m + ":\n"
            for f in m_:
                error += "\t" + f + "\n"

        raise click.exceptions.BadParameter(error, param_hint="county-configuration")

    return conf


def ensure_location_designation_present(county):
    if "locationDesignations" not in county:
        county["locationDesignations"] = [
            EssentialDesignationExtractor().from_county(county)
        ]
    updated = []
    for locdes in county["locationDesignations"]:
        path, _ = make_file_absolute(os.curdir, locdes)
        updated.append(path)
    county["locationDesignations"] = updated
    return county


def test_code_available(java_location: os.PathLike) -> None:
    """
    Do a quick test to see if PanSim is available and accessible
    Args:
            java_location: Location of Java executable on this system

    Throws a bad argument exception if PanSim is not working
    """
    try:
        result = subprocess.run(
            [java_location, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        print("Using", str(result.stdout.splitlines()[0], "utf-8"))
    except FileNotFoundError as e:
        raise click.exceptions.BadParameter(
            f"The provided Java binary {java_location} could not be used to start a Java VM",
            param_hint="--java",
        )

    result = subprocess.run(
        ["pansim", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    if result.returncode == 0:
        print("Pansim is available")
    else:
        raise click.exceptions.BadArgumentUsage(
            f"PanSim was not found. Please install PanSim before attempting calibration"
        )


def get_expected_end_date(toml_county_configuration: str, num_steps: int) -> datetime:
    """
    Get the expected date of the last simulation date
    Args:
            toml_county_configuration: Location of county configuration file
            num_steps: Number of steps this simulation runs

    Returns:
            String object of date of last expected simulation day
    """
    config = toml.load(toml_county_configuration)
    last_date = config["simulation"]["startdate"] + timedelta(days=num_steps)
    return last_date


def make_fips_long(county_fips: int or str, state_fips: int or str = 51) -> int:
    """
    Given a county fips code, return the combined state and county fips code
    Args:
        county_fips:    The FIPS code of the state (e.g., VA has fips code 51)
        state_fips:     The FIPS code of the county within the state

    Returns: Combined state and county fips code
    """
    state_fips = str(state_fips) + "0" * (5 - len(str(state_fips)))
    return int(state_fips[: -1 * len(str(county_fips))] + str(county_fips))


def int_list_to_int(lst: IntArray, largest_int: int) -> int:
    return int("".join(["{0:0{1}b}".format(x, len(f"{largest_int:b}")) for x in lst]), 2)


def int_to_int_list(number: hex, largest_int: int, array_size: int) -> IntArray:
    byte_size = len(f"{largest_int:b}")
    n = bin(number)[2:]
    n = "0" * ((array_size * byte_size) - len(n)) + n
    numbers = [n[a:a + byte_size] for a in range(0, len(n), byte_size)]
    return [int(number, 2) for number in numbers]


def test_int_list_serialization(n_repeats=100, max_size=10, list_range=(5, 20)):
    for _ in range(n_repeats):
        _lst = [random.randint(0, max_size) for _ in range(random.randint(*list_range))]
        m = max(_lst)
        number = int_list_to_int(_lst, m)
        result = int_to_int_list(number, m, len(_lst))
        assert result == _lst, f"Expected list reduced to {number} to be {_lst}, got {result}"


alphabet = "".join(map(str, range(10))) + string.ascii_lowercase + string.ascii_uppercase
alphabet_reverse = dict((c, i) for (i, c) in enumerate(alphabet))


def base_encode(number, base=36):
    if not isinstance(number, int):
        raise TypeError(f'Number {number} must be an integer!')
    if base > len(alphabet):
        raise ValueError(f"The highest base supported is {len(alphabet)}, got {base}")

    base_string = ''

    while number:
        number, i = divmod(number, base)
        base_string = alphabet[i] + base_string

    return base_string or alphabet[0]


def base_decode(number: str, base=36):
    n = 0
    for c in number:
        n = (n * base + alphabet_reverse[c])
    return n


def test_base_encoding():
    print(len(alphabet), alphabet)
    for base in range(2, len(alphabet) + 1):
        print(f"Testing base {base}")
        for number in range(100000):
            encoded = base_encode(number, base)
            decoded = base_decode(encoded, base)
            assert number == decoded, f"Encoded number {number} in base {base} resulted in {encoded}, which decodes to {decoded}"


def get_project_root(*path):
    return os.path.join(Path(__file__).parent.parent, *path)


if __name__ == "__main__":
    test_int_list_serialization(100, 100, list_range=(1, 1000))
    test_base_encoding()
