"""
For initial rounds of calibration, we have used the case data provided by USA Facts.
The dataset was accessed in September 2020.
Lack of testing infrastructure means the number of reported cases likely was underreported, but it is unclear if these
numbers can be scaled linearly over time, which is only the case if the testing capacity increased linearly with the
actual number of infections.

Another

Alternative sources for case data exist:

* HealthData.gov has provided a dataset of all tests and positivity rate officially recorded:
    https://healthdata.gov/dataset/COVID-19-Diagnostic-Laboratory-Testing-PCR-Testing/j8mb-icvb
* The Covid Tracking Project has a similar dataset, going until March 7, 2021, but only on the state level
    https://covidtracking.com/data/download
* Based on the data of The Covid Tracking Project, Youyang Gu has developed a dataset of estimations of actual number
    of infections, which somehow includes individual counties
    https://covid19-projections.com/

The purpose of this Python script is to compare the three datasets, find out if recorded case counts correspond
(consider e.g., https://covidtracking.com/analysis-updates/federal-covid-19-test-data-is-getting-better), if the
positivity rate of tests can be used as a proxy for actual number of cases, and to try and replicate the calculations
from the covid19 tracking project.

We restrict analysis to the first 9 months of 2020, and only look at Virginia.
"""
import math
import os
import shutil
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from termcolor import colored

from utility.utility import get_project_root

calibration_counties = [51540, 51065, 51109, 51075]


def aggregate_county_level_data_to_state():
    new_dictionary = dict()
    for dataset, information in datasets.items():
        if information['countylevel']:
            new_dictionary[dataset] = information.copy()
            new_dictionary[dataset]['data'] = _aggregate_single_dataset_to_state(information['data'])
        else:
            new_dictionary[dataset] = information

    return new_dictionary


def _aggregate_single_dataset_to_state(data):
    aggregated = dict()
    for fips, dates in data.items():
        for date, records in dates.items():
            if date not in aggregated:
                aggregated[date] = dict()
            for key, val in records.items():
                if isinstance(val, str):
                    continue
                try:
                    aggregated[date][key] += val
                except KeyError:
                    aggregated[date][key] = val

    return aggregated


def get_state_level_infected_from_data():
    infected = defaultdict(dict)
    deaths = defaultdict(dict)
    state_level_data = aggregate_county_level_data_to_state()
    for date, values in state_level_data['baseline']['data'].items():
        for dataset, information in state_level_data.items():
            if information['deaths_key'] is not None:
                (n_cases, n_death) = get_cases_and_deaths(information, None, date)
                infected[date][dataset] = n_cases
                deaths[date][dataset] = n_death

    return infected, deaths


def make_time_series_plot(x: List, values: List[Tuple[str, list]], title: str or None = None, callback = None):
    fig, ax = plt.subplots()
    lns = list()
    for name, value_list in values:
        lns += (ax.plot(x, value_list, label=name))
    if title is not None:
        plt.title(title)
    plt.xticks(x, rotation=60)
    ax.xaxis.set_major_locator(MultipleLocator(7))
    if callback is not None:
        lns += callback((fig, ax))

    plt.legend(lns, [l.get_label() for l in lns])
    plt.tight_layout()
    if title is not None:
        plt.savefig(title.lower().replace(" ", "_") + ".png")
    plt.show()


def analyse_state_level_infected_differences(values, print_intermediate_results=False):
    num_errors, differences = 0, list()
    for date, dataset in values.items():
        if not __all_equal(dataset):
            num_errors += 1
            grouped_values = defaultdict(list)
            for name, value in dataset.items():
                grouped_values[value].append(name)
            occurring_values = sorted(list(grouped_values.keys()), key=lambda x: len(grouped_values[x]))
            if print_intermediate_results:
                print(colored("\t" + date, 'red'), occurring_values[0], end=': ')
                for val in occurring_values[1:]:
                    print(colored(f'{val} ({",".join(grouped_values[val])})'), end=', ')
                print("")
            differences.append(max(occurring_values) - min(occurring_values))
        elif print_intermediate_results:
            print(colored(f"\t{date}: {list(dataset.values())[0]}", 'green'))

    print(f"Maximum difference was {np.nanmax(differences)} across {num_errors} errors, average error was {np.average([x for x in differences if not np.isnan(x)])}")


def __all_equal(lst):
    if isinstance(lst, dict):
        lst = list(lst.values())
    return all([x == lst[0] for x in lst[1:]])


def get_county_level_infected_from_data():
    """
    Creates a dataset of data sets reporting only the cumulative number of cases and deaths in each county.
    Only data sets that report on the county level are included.

    Finally, the difference in reported values is compared and printed to the terminal
    """
    infected = defaultdict(lambda: defaultdict(dict))
    deaths = defaultdict(lambda: defaultdict(dict))

    for fips, dates in datasets['baseline']['data'].items():
        for date, records in dates.items():
            for dataset, information in datasets.items():
                if information['countylevel']:
                    (n_cases, n_death) = get_cases_and_deaths(information, fips, date)
                    infected[fips][date][dataset] = n_cases
                    deaths[fips][date][dataset] = n_death

    print("Verifying cases consistency")
    print(assert_values_equal(infected), "errors found")

    # print("Verifying deaths consistency")
    # print(assert_values_equal(deaths), "errors found")


def assert_values_equal(dictionary):
    """
    This method serves to find the differences in reported (cumulative) case counts in data sets
    passed in the dictionary.

    Summary of results is printed to the terminal

    Args:
        dictionary: The dictionary containing all the loaded data sets

    """
    counties_with_errors, counties_total = 0, 0
    max_delta_total = 0
    correct_counties, incorrect_counties = list(), list()
    for county, county_data in dictionary.items():
        counties_total += 1
        errors, total, max_delta = 0, 0, 0
        first_day_error, last_day_error, difference = None, None, 0
        for date, values_to_compare in county_data.items():
            values = list(values_to_compare.values())
            total += 1
            if not __all_equal(values):
                errors += 1
                first_day_error = date if first_day_error is None else first_day_error
                last_day_error = date
                differences = [abs(a - b) for a, b in zip(values, values[1:])]
                difference += sum(differences)
                max_delta = max(max_delta, max(differences))
        if errors > 0:
            max_delta_total = max(max_delta_total, max_delta)
            counties_with_errors += 1
            incorrect_counties.append(county)
            if county in calibration_counties:
                print(f"\tCounty {county} has {errors} dates that do not correspond out of {total}")
                print(f"\t\tThe difference was {difference} and occurred between {first_day_error} and {last_day_error}. Max difference on any given day was {max_delta}")
        else:
            correct_counties.append(county)
    print(f"{counties_with_errors} out of {counties_total} did not correspond completely. Maximum difference on any given day was {max_delta_total}")


def get_cases_and_deaths(dataset_information, fips, date):
    """
    Extract the cumulative number of cases and deaths from a given data set, based on the keys specified for those
    values
    Args:
        dataset_information: The dictionary containing the data set
        fips: The FIPS code for which to extract the values
        date: The date for which to extract the values

    Returns:
        Tuple of cumulative (cases, deaths) reported by the data set for the given county at the given date

    """
    cases, deaths = 0, 0
    to_extract_from = dataset_information['data'] if fips is None else dataset_information['data'][fips]
    if date in to_extract_from:
        cases = to_extract_from[date][dataset_information['cases_key']]
        deaths = to_extract_from[date][dataset_information['deaths_key']]

    return cases, deaths


def merge_covid_projection_datasets(projection_dataset, estimates_dataset):
    """
    Two data sets for the covid19 projections exist, with a lot of overlap. However, some keys appear in only one,
    and one data set has a higher precision recorded for certain values. This method merges them into one

    Args:
        projection_dataset: The data set containing the COVID-19 projections
        estimates_dataset:  The data set containing the estimation of actual COVID-19 infections

    Returns:

    """
    print("Combining projections and estimations datasets")
    combined_data = projection_dataset
    for fips, data in estimates_dataset.items():
        for date, records in data.items():
            for record, value in records.items():
                if record not in combined_data[fips][date]:
                    combined_data[fips][date][record] = value
                elif value != combined_data[fips][date][record]:
                    if math.isnan(combined_data[fips][date][record]):
                        combined_data[fips][date][record] = value
                    elif isinstance(value, float) and isinstance(combined_data[fips][date][record], float):
                        more_precise, less_precise, same = get_more_precise_number(value, combined_data[fips][date][record])
                        if same:
                            combined_data[fips][date][record] = more_precise
                        else:
                            print(f"[WARNING] Cannot resolve difference in values of FIPS {fips} at {date}: {value} vs "
                                  f"{combined_data[fips][date][record]}")
    return combined_data, True


def get_more_precise_number(n1: float, n2: float, debug=False):
    """
    Tests if two floats with different precision (because parsed from string) are the same to the most significant common
    decimal

    Args:
        n1: Float 1 to compare
        n2: Float 2 to compare
        debug: Print values that are compared if significance differs

    Returns:
        tuple with the most precise float at position
        0, the less precise float at position 1, and a boolean at position 2 that is true iff n1 and n2 are
        the same -- accounted for accuracy -- and False if they are not
    """
    if n1 == n2:
        return n1, n2, True
    elif len(str(n2)) > len(str(n1)):
        return get_more_precise_number(n2, n1)
    else:
        precision = get_precision(n2)
        n1_s = f"{n1:.{precision}f}"
        n2_s = f"{n2:.{precision}f}"
        if debug:
            print(precision)
            print(n1_s, "\t", f"{n1:f}", "\t", n1)
            print(n2_s, "\t", f"{n2:f}", "\t", n2)
        return n1, n2, n1_s == n2_s


def get_precision(f: float):
    """
    Get the precision or significance of a float value
    Args:
        f: Float value

    Returns:
        Significance of float value, i.e., number of digits after the decimal place

    """
    f_str = str(f).lower()
    precision_bonus = 0
    if "e" in f_str:
        f_str, precision_bonus = f_str.split("e")

    main, digits = f_str.split(".")
    precision = len(digits)
    return precision - 1 + (-1 * (int(precision_bonus) - 1))


def read_data(dataset):
    """
    Reads a data set that contains data split by state and by date and, optionally, by county FIPS code

    Args:
        dataset: The path to a dataset

    Returns:
        A nested dictionary, where the first key is the state code, the second key is the FIPS code of the individual
        county (if present), the last key is a date, and the value is a dictionary encoding a line in the CSV
    """
    print("Loading", dataset)
    with open(dataset, 'r') as data_in:
        headers = [fix_type(x) for x in data_in.readline()[:-1].split(",")]
        state_index = headers.index('state')
        include_fips = 'fips' in headers
        data = defaultdict(dict) if include_fips else dict()
        for line in data_in:
            line = [fix_type(x) for x in line[:-1].split(",")]
            if line[state_index].lower() in ['va', 'virginia']:
                line_data = dict(zip(headers, line))  # TODO verify this works for all data sets
                if include_fips:
                    data[line_data["fips"]][line_data['date']] = line_data
                else:
                    data[line_data['date']] = line_data

    return data, include_fips


def read_testing_data(dataset):
    """
    Normalize the testing dataset to one record per day
    Args:
        dataset: The testing data set to load

    Returns:
        Normalized dataset with one record per day
    """
    print("Loading testing dataset")
    data = defaultdict(dict)
    with open(dataset, 'r') as data_in:
        headers = [fix_type(x) for x in data_in.readline()[:-1].split(",")]
        state_index = headers.index('state')
        for line in data_in:
            line = [fix_type(x) for x in line[:-1].split(",")]
            if line[state_index] == 'VA':
                line = dict(zip(headers, line))
                date = line['date'].replace("/", "-")
                data[date][f'new_results_reported_{line["overall_outcome"].lower()}'] = line['new_results_reported']
                data[date][f'total_results_reported_{line["overall_outcome"].lower()}'] = line['total_results_reported']

    return data, False


def fix_type(value: str):
    """
    Tries to convert a string value to the appropriate type, e.g. a number or a float
    Args:
        value: String with value

    Returns:
        Value in guessed type
    """
    value = value[1:-1] if value.startswith('"') and value.endswith('"') else value
    sign = -1 if value.startswith("-") else 1
    value = value[1:] if value.startswith("-") else value

    if value == '':
        return math.nan
    elif value.isdecimal():
        return sign * int(value)

    try:
        return sign * float(value)
    except ValueError:
        return value if sign == 1 else "-" + value


def load_data_sets():
    for _dataset, _information in datasets.items():
        _information['data'], _information['countylevel'] = _information['reader'](_information['path'])
        print(_dataset, _information['countylevel'])

    # The following two data sets have been merged and can be removed
    datasets.pop('projections')
    datasets.pop('estimates')


def load_single_dataset(dataset):
    return datasets[dataset]['reader'](datasets[dataset]['path'])


datasets = dict(
        baseline=dict(
            path=os.path.join(get_project_root(), "external", "va-counties-covid19-cases.csv"),
            reader=read_data,
            url=None,
            cases_key="cases",
            deaths_key="deaths"
        ),
        projections=dict(
            path=os.path.join("data", "covid-projections", "covid19-projections.csv"),
            reader=read_data,
            url="https://raw.githubusercontent.com/youyanggu/covid19-infection-estimates-latest/main/counties/latest_VA.csv",
            cases_key="total_cases",
            deaths_key="total_deaths"
        ),
        estimates=dict(
            path=os.path.join("data", "covid-projections", "us_counties_infection_estimates_time_series.csv"),
            reader=read_data,
            url=None,
            cases_key="total_cases",
            deaths_key="total_deaths"
        ),
        tracking=dict(
            path=os.path.join("data", "covidtracking.com", "virginia-history.csv"),
            reader=read_data,
            url="https://covidtracking.com/data/download/virginia-history.csv",
            cases_key="positive",
            deaths_key="death"
        ),
        testing=dict(
            path=os.path.join("data", "healthdata.gov", "COVID-19_Diagnostic_Laboratory_Testing__PCR_Testing__Time_Series.csv"),
            reader=read_testing_data,
            url="https://healthdata.gov/dataset/COVID-19-Diagnostic-Laboratory-Testing-PCR-Testing/j8mb-icvb",
            cases_key="total_results_reported_positive",
            deaths_key=None
        ),
        combined_projections=dict(
            path=None,
            reader=lambda _: merge_covid_projection_datasets(datasets['projections']['data'], datasets['estimates']['data']),
            url=None,
            cases_key="total_cases",
            deaths_key="total_deaths"
        )
    )


def compare_case_data():
    load_data_sets()
    get_county_level_infected_from_data()
    state_level_infected, state_level_deaths = get_state_level_infected_from_data()
    infected_x = list(sorted(state_level_infected.keys()))
    make_time_series_plot(
        infected_x,
        [(name, [state_level_infected[date][name] for date in infected_x]) for name in state_level_infected[infected_x[0]]],
        "State Level Infected"
    )

    deaths_x = list(sorted(state_level_deaths.keys()))
    make_time_series_plot(
        deaths_x,
        [(name, [state_level_deaths[date][name] for date in deaths_x]) for name in state_level_deaths[deaths_x[0]]],
        "State Level Deaths"
    )

    analyse_state_level_infected_differences(state_level_infected)
    analyse_state_level_infected_differences(state_level_deaths)


def compare_confirmed_positive_tests_with_baseline():
    baseline, _ = load_single_dataset('baseline')
    testing, _ = load_single_dataset('testing')
    aggregated = _aggregate_single_dataset_to_state(baseline)
    dates = sorted(list(aggregated.keys()))
    cases = [aggregated[date]['cases'] for date in dates]
    positive = [testing[date]['total_results_reported_positive'] for date in dates]
    p = 'new_results_reported_positive'
    n = 'new_results_reported_negative'
    percentages = [testing[date][p] / (testing[date][n] + testing[date][p]) if testing[date][p] + testing[date][n] != 0 else np.nan for date in dates]

    make_time_series_plot(
        dates,
        [('cumulative cases (calibration)', cases), ('positive tests (HealthData.gov)', positive)],
        "Confirmed Cases vs Positive Tests",
        lambda x: __add_plot_on_second_axes(dates, percentages, x, 'Positivity rate', True)
    )


def stack_recorded_cases_in_plot():
    baseline, _ = load_single_dataset('baseline')
    dates = sorted(list(baseline[calibration_counties[0]].keys()))
    testing, _ = load_single_dataset('testing')
    cases = [sum([baseline[fips][date]['cases'] for fips in calibration_counties if date in baseline[fips]]) for date in dates]
    all_cases = [_aggregate_single_dataset_to_state(baseline)[x] for x in dates]

    p = 'new_results_reported_positive'
    n = 'new_results_reported_negative'

    percentages = [
        testing[date][p] / (testing[date][n] + testing[date][p]) if testing[date][p] + testing[date][n] != 0 else np.nan
        for date in dates]

    for key, title, case_data in [('new', "Daily", cases), ('total', "Total", all_cases)]:
        positive = [testing[date][f'{key}_results_reported_positive'] for date in dates]
        negative = [testing[date][f'{key}_results_reported_negative'] for date in dates]
        fig, ax = plt.subplots()
        lns = ax.stackplot(dates, positive, negative, labels=["Positive", "Negative"])
        scale_factor = int((max(positive) + max(negative)) / max(cases))
        lns += ax.plot(dates, [c*scale_factor for c in cases], label=f"Cumulative Cases * {scale_factor} (calibration)")
        plt.xticks(dates, rotation=60)
        ax.xaxis.set_major_locator(MultipleLocator(7))
        plt.title(f"{title} Test Results (Virginia)")
        lns += __add_plot_on_second_axes(dates, percentages, (fig, ax), 'Positivity rate', True)
        plt.legend(lns, [l.get_label() for l in lns], loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{title}_test_results_virginia.png")
        plt.show()


def __add_plot_on_second_axes(dates, percentages, params, label, format_as_percentages=False):
    fig, ax = params
    ax2 = ax.twinx()
    ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
    lns = ax2.plot(dates, percentages, label=label)
    smooth = __get_n_days_sliding_window(percentages, 7)
    ax2.tick_params(axis='y', labelcolor=lns[0].get_color())
    lns += ax2.plot(dates, smooth, label=f'{label} (7-day sliding window avg)')
    if format_as_percentages:
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:0.0%}'.format(y)))
    return lns


def __get_n_days_sliding_window(data, days):
    smooth = list()
    for i in range(len(data)):
        if i < days:
            smooth.append(np.nan)
        else:
            smooth.append(sum(data[i - days:i]) / len(data[i - days:i]))
    return smooth


def plot_experiment_with_projected():
    cases = _load_experiment_epicurve()
    projections, _ = load_single_dataset('projections')
    dates = sorted(list(cases.keys()))

    extractor = lambda x: [sum([projections[county][date][x] for county in calibration_counties]) for date in dates]

    cases_data = dict(
        mean=[np.average(cases[x]) for x in dates],
        lower=[np.average(cases[x]) - np.std(cases[x]) for x in dates],
        upper=[np.average(cases[x]) + np.std(cases[x]) for x in dates]
    )
    projections_data = dict(
        mean=extractor('total_infected_mean'),
        lower=extractor('total_infected_lower'),
        upper=extractor('total_infected_upper')
    )
    projections_data_scaled = dict(
        mean=[x * 4 for x in projections_data['mean']],
        lower=[x * 4 for x in projections_data['lower']],
        upper=[x * 4 for x in projections_data['upper']]
    )
    total_cases = extractor('total_cases')
    total_cases_scaled = [x * 36 for x in total_cases]
    total_cases_data = dict(mean=total_cases, lower=total_cases, upper=total_cases)
    total_cases_scaled_data = dict(mean=total_cases_scaled, lower=total_cases_scaled, upper=total_cases_scaled)

    fig, ax = plt.subplots()
    for title, data in [
        ('infected (simulated)', cases_data),
        ('real world projection', projections_data),
        ('projection (* 4)', projections_data_scaled),
        ('confirmed', total_cases_data),
        ('confirmed x 36', total_cases_scaled_data)
    ]:
        line = ax.plot(dates, data['mean'], label=title)[0]
        ax.fill_between(dates, data['lower'], data['upper'], alpha=0.2, facecolor=line.get_color(), antialiased=True)

    plt.title("Projected vs. Simulated (cumulative)")
    plt.legend()
    plt.xticks(dates, rotation=60)
    ax.xaxis.set_major_locator(MultipleLocator(7))
    plt.tight_layout()
    plt.savefig('project-v-simulated.png')
    plt.show()


def _load_experiment_epicurve():
    dir = os.path.join('data', 'calibrated-extended-experiment')
    cases = defaultdict(list)
    for f in os.listdir(dir):
        with open(os.path.join(dir, f)) as cases_in:
            headers = cases_in.readline()[:-1].split(";")
            for line in cases_in:
                values = line[:-1].split(";")
                cases[values[headers.index('Date')]].append(
                    sum([int(values[headers.index(x)]) for x in
                         ['EXPOSED', 'INFECTED_SYMPTOMATIC', 'INFECTED_ASYMPTOMATIC', 'RECOVERED']])
                )

    return cases


def plot_disease_calibration_with_prevalence_ratio():
    positivity_rates = get_positivity_rates()
    baseline = load_single_dataset('projections')[0]
    exp = _load_experiment_epicurve()
    dates = sorted(list(exp.keys()))

    cases = dict(zip(dates, [sum([baseline[fips][date]['total_cases'] for fips in calibration_counties if date in baseline[fips]]) for date in dates]))


    scaled_rates = [cases[date] * _prevalence_ratio(positivity_rates, date) for date in dates]
    w_prev_ratio = [cases[x] * _prevalence_ratio(positivity_rates, x) * 6 for x in dates]
    smooth_w_prev = __get_n_days_sliding_window(w_prev_ratio, 7)

    make_time_series_plot(
        dates,
        [
            ('experiment', [np.average(exp[x]) for x in dates]),  # TODO, plot uncertainty interval later
            # ('cases', [cases[x] for x in dates]),
            ('cases * 36', [cases[x] * 36 for x in dates]),
            # ('scaled w/ prevalence', scaled_rates),
            # TODO, smooth?
            # TODO, Any way to mathematically arrive at 0.16?
            ('cases * prevalence * 6', w_prev_ratio),
            ('cases * prevalence * 6 (7-day sliding window avg)', smooth_w_prev),
        ],
        title="Testing Use of Prevalence Index",
        callback=lambda x: __add_plot_on_second_axes(
            dates,
            [_prevalence_ratio(positivity_rates, date) for date in dates],
            x,
            "Prevalence Ratio"
        )
    )


def get_positivity_rates():
    positivity_rate = dict()
    for date, information in load_single_dataset("testing")[0].items():
        total = information['new_results_reported_positive'] + information['new_results_reported_negative']
        positivity_rate[date] = 0 if total == 0 else information['new_results_reported_positive'] / total
    return positivity_rate


def _prevalence_ratio(positivity_rates, date):
    day_i = (datetime.strptime(date, "%Y-%m-%d") - datetime(2020, 2, 12)).days
    rate = positivity_rates[date] if date in positivity_rates else 0
    return (1500 / (day_i + 50)) * (rate**0.5) + 2


def todo():
    """
    Based on the analysis in this script, we have decided that using the estimated true number of infections
    provided by the COVID19 Tracking Project is a suitable method of removing the scaling factor from our current
    calibration process.

    These estimations use the confirmed number of cases as their base, which we have seen previously,
    is lower than the number of positive tests. This raises the question how cases are confirmed if not through positive
    testing, and further raises the question what would happen if we apply the same prevalence ratio to the number
    of positive tests, instead of to the number of confirmed cases.

    That is what we are here to find out
    """
    verify_prevalence_algorithm()


def verify_prevalence_algorithm():
    # projections, _ = load_single_dataset('projections')
    tracking, _ = load_single_dataset('tracking')
    testing, _ = load_single_dataset('testing')
    dates = list(sorted(tracking.keys()))
    for date in dates:
        if date in testing:
            print(date, tracking[date]['positiveIncrease'], testing[date]['new_results_reported_positive'])
        # print(testing[date])
        # if not data['positive'] == data['positiveCasesViral'] + data['probableCases']:
        #     print(date, data['positive'], data['positiveCasesViral'], data['probableCases'])


def rewrite_to_known_format():
    load_data_sets()
    known_headers = ["date", "county", "state", "fips", "cases", "deaths"]
    known_format = defaultdict(lambda: defaultdict(dict))
    for fips, dates in datasets['combined_projections']['data'].items():
        for date, values in dates.items():
            if date < "2021-02-21" and not np.isnan(values["total_infected_mean"]):
                assert len(str(fips)) == 5
                known_format[date][fips] = dict(
                    date=date,
                    county=values["county"],
                    state="Virginia",
                    fips=fips,
                    cases=values["total_infected_mean"],
                    deaths=values["total_deaths"]
                )

    outfile = os.path.join(get_project_root(), "external", "va-counties-estimated-covid19-cases.csv")
    with open(outfile, 'w') as csv_out:
        csv_out.write(",".join(known_headers) + "\n")
        dates = sorted(list(known_format.keys()))
        for date in dates:
            for fips, data in known_format[date].items():
                csv_out.write("{date},{county},{state},{fips},{cases},{deaths}\n".format(**data))

    print("Created", outfile)


if __name__ == "__main__":
    # compare_case_data()
    # compare_confirmed_positive_tests_with_baseline()
    # stack_recorded_cases_in_plot()
    # plot_experiment_with_projected()
    # plot_disease_calibration_with_prevalence_ratio()

    # todo()
    rewrite_to_known_format()
"""
TODO:

Plot, in 4 counties:
    Case data, scaled by 36 (calibrated) from experiment
    Projected data
    Positivity rate
"""