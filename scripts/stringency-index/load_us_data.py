import json
import os.path
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from matplotlib.ticker import MultipleLocator
from termcolor import colored

from OxCGRTUS_utils import assert_excel_compatible_or_convert_to_ods, store_as_json, save_excel_file_from_github

# See https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/index_methodology.md#calculating-sub-index-scores-for-each-indicator
indicators = dict(
    C1=dict(max=3, flag=1),
    C2=dict(max=3, flag=1),
    C3=dict(max=2, flag=1),
    C4=dict(max=4, flag=1),
    C5=dict(max=2, flag=1),
    C6=dict(max=3, flag=1),
    C7=dict(max=2, flag=1),
    C8=dict(max=4, flag=0),
    E1=dict(max=2, flag=1),
    E2=dict(max=2, flag=0),
    H1=dict(max=2, flag=1),
    H2=dict(max=3, flag=0),
    H3=dict(max=2, flag=0),
    H6=dict(max=4, flag=1),
    H7=dict(max=5, flag=1),
    H8=dict(max=3, flag=1),
)

def load_virginia_stringency_data(norm_file=None):
    filename_base='OxCGRTUS_timeseries_all'
    if os.path.isfile(f'{filename_base}.json'):
        with open(f'{filename_base}.json', 'r') as json_in:
            values = json.load(json_in)
    else:
        filename = assert_excel_compatible_or_convert_to_ods(filename_base)
        values = load_virginia_stringency_as_dictionary(filename)
        if norm_file is not None:
            values = add_activate_norms_to_data(values, norm_file)
        store_as_json(values, filename_base)

    return values


def load_more_data():
    filename = 'OxCGRT_US_latest.csv'
    save_excel_file_from_github(filename=filename)
    df = pd.read_csv(filename)
    virginia_df = df[df['RegionCode'] == 'US_VA']
    notes_column = list(filter(lambda x: 'Notes' in x, virginia_df.columns))
    virginia_notes_df = virginia_df[['Date'] + notes_column]
    for row in virginia_notes_df.iterrows():
        date = row[1][0]
        if date > 20200628:
            break
        without_date = row[1][1:]
        if not all(map(lambda x: pd.isna(x), without_date)):
            date_comments = list(filter(pd.notna, row[1]))
            print(datetime.strptime(f'{date}', "%Y%m%d").strftime("%Y-%m-%d"))
            for comment in date_comments[1:]:
                flag = notes_column[list(row[1]).index(comment)-1]
                # if flag.startswith("C"):
                print(f"\t - [{flag}]  " + comment)
            print("\n\n")


def verify_metrics_for_stringency():
    filename = 'OxCGRT_US_latest.csv'
    save_excel_file_from_github(filename=filename)
    df = pd.read_csv(filename)
    virginia_df = df[df['RegionCode'] == 'US_VA']

    # Apart from the last value, si equals sidp, and sil equals sildp (last value is NaN for the non-display lists)
    # Flags are either 0, 1 or NaN. Can ignore?

    values = [calculate_index(row) for row in virginia_df.to_dict(orient="records")]

    scale = virginia_df['StringencyIndex'].values[130] / values[130]
    print("Scale factor:", scale)

    target = virginia_df['StringencyIndex'].values
    target = [0 if np.isnan(x) else x for x in target]

    for (x, y) in zip(values, target):
        if x != y:
            print(x, y)

    print("RMSE", sklearn.metrics.mean_squared_error(values, target, squared=False))

    dates = virginia_df['Date'].values
    xvalues = np.arange(0, len(dates), 1)
    fig, ax = plt.subplots()
    ax.plot(xvalues, virginia_df['StringencyIndex'].values, label='StringencyIndex')
    ax.plot(xvalues, list(map(lambda x: x * 1, values)), label="Sum")
    plt.legend()
    plt.show()


def get_dates_and_notes_for_stringency():
    filename = 'OxCGRT_US_latest.csv'
    save_excel_file_from_github(filename=filename)
    df = pd.read_csv(filename)
    virginia_df = df[df['RegionCode'] == 'US_VA']

    index_indicators = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "StringencyIndex", "Date"]
    last_stringency = 0
    for row in virginia_df.to_dict(orient="records"):
        if row['Date'] > 20200629:
            return
        keys = list(row.keys())
        for key in keys:
            if not any(map(lambda x: key.startswith(x), index_indicators)):
                row.pop(key)
        if row["StringencyIndex"] != last_stringency:
            last_stringency = row["StringencyIndex"]
            print(row["Date"], last_stringency)

            notes_dct = dict()
            for key in row.keys():
                if "Notes" in key and pd.notna(row[key]):
                    notes_dct[key] = row[key]
            for note_key, note in notes_dct.items():
                print("\t", note_key, note)

            print("\n")


def calculate_index(row, index_indicators=dict(C=[1, 2, 3, 4, 5, 6, 7, 8], H=[1])):
    """
    Calculates the index according to
    https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/index_methodology.md#methodology-for-calculating-indices

    Args:
        row: A row from the data frame converted to a dictionary
        index_indicators: Dictionary with component letters as key, and list of
        component numbers associated with that letter as value that are used to
        calculate this index (default argument is for stringency index)

    Returns:
        The index calculated from the row
    """
    n_component_indicators = 0
    index = 0
    for indicator_letter in index_indicators:
        for indicator_number in index_indicators[indicator_letter]:
            n_component_indicators += 1
            indicator_name = f"{indicator_letter}{indicator_number}"
            flag_value = row[f"{indicator_name}_Flag"] if indicators[indicator_name]["flag"] != 0 else 0
            indicator_value_column = next(filter(lambda x: x.startswith(indicator_name) and not "Notes" in x and not "Flag" in x, row.keys()))
            indicator_value = row[indicator_value_column]
            sis = sub_index_score(indicator_name, indicator_value, flag_value)
            if not np.isnan(sis):
                index += sis

    return round((1 / n_component_indicators) * index, 2)


def sub_index_score(indicator: str, indicator_val: int, flag_val: int):
    """
    See https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/index_methodology.md#calculating-sub-index-scores-for-each-indicator
    Args:
        indicator: Name of the indicator
        indicator_val: the recorded policy value on the ordinal scale (vj,t)
        flag_val: the recorded binary flag for that indicator (fj,t)
    
    Returns:
        Sub index score
    """
    # https://camo.githubusercontent.com/39e4a6c21e91f0386a8e8f1760826eac7a1526271b5fb8936515d0d40618e6d3/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f706e672e6c617465783f253238322532392535437171756164253230495f2537426a2532437425374425334431303025354366726163253742765f2537426a253243742537442d302e35253238465f2537426a2537442d665f2537426a253243742537442532392537442537424e5f2537426a253744253744
    # 100 * ((vjt - (0.5 * (Fj - fjt) )) / NJ)
    return 100 * ((indicator_val - (0.5 * (indicators[indicator]["flag"] - flag_val))) / indicators[indicator]["max"])


def load_virginia_stringency_as_dictionary(filename):
    """
    Extracts all the values for Virginia into a dictionary
    """
    # Load data from downloaded (and converted?) file
    sheet_to_df_map = pd.read_excel(filename, sheet_name=None)

    # Create a dictionary for all the values for Virginia
    values = defaultdict(dict)
    for sheet, df in sheet_to_df_map.items():
        print("Reading sheet", sheet)
        is_virginia = df['region_code'] == 'US_VA'
        virginia_df = df[is_virginia]
        for column in virginia_df:
            try:
                date_object = datetime.strptime(column, '%d%b%Y')
                value = virginia_df[column].values[0]
                values[datetime.strftime(date_object, "%Y-%m-%d")][sheet] = float(value)
            except ValueError:
                print(f"Column '{column}' is not a date. Skipping")

    return values


def add_activate_norms_to_data(values, norms_file):
    norm_events = load_norm_events(norms_file)
    active_norms = list()
    dates = sorted(list(values.keys()))
    for date in dates:
        for activated in norm_events[date]['activated']:
            active_norms.append(activated)
        for deactivated in norm_events[date]['deactivated']:
            active_norms.remove(deactivated)
        print(date, len(active_norms))
        values[date]['norms'] = list(active_norms)

    return values


def load_norm_events(norms_file):
    event_dict = defaultdict(lambda: {'activated': [], 'deactivated': []})
    with open(norms_file, 'r') as norms_in:
        for line in norms_in:
            data = split_norm_line(line)[:4]
            if data[1] != '':
                norm_params = list(map(lambda x: x.replace(' ', ''), split_norm_line(data[3])))
                event_dict[data[0]]['activated'].append((data[1], norm_params))
                event_dict[data[2]]['deactivated'].append((data[1], norm_params))

    return event_dict


def split_norm_line(norm_line):
    data = list()
    in_cell = False
    accumulator = ""
    for char in norm_line:
        if char == "," and not in_cell:
            data.append(accumulator)
            accumulator = ""
        elif char == '"':
            in_cell = not in_cell
        else:
            accumulator += char
    data.append(accumulator)
    return data


def find_first_important_date(values):
    dates = sorted(list(values.keys()))
    stringency_found = False
    norm_found = False
    for date in dates:
        if not stringency_found and values[date]['stringency_index'] != 0:
            print(date, values[date]['stringency_index'])
            stringency_found = True
        if not norm_found and len(values[date]['norms']) > 0:
            print(date, values[date]['stringency_index'])
            norm_found = True
        if stringency_found and norm_found:
            return


def plot_number_of_norms_vs_stringency(values):
    fig, ax = plt.subplots()
    x_labels = sorted(filter(lambda x: x < '2020-06-28', list(values.keys())))
    n_norms = list(map(lambda x: len(values[x]['norms']), x_labels))
    n_norms_normalized = list(map(lambda x: x / max(n_norms) * 100, n_norms))
    stringency = list(map(lambda x: values[x]['stringency_index'], x_labels))
    normalized_stringency = list(map(lambda x: x / max(stringency) * 100, stringency))
    x = np.arange(0, len(x_labels), 1)
    ax.plot(x_labels, n_norms_normalized, normalized_stringency)
    plt.ylabel("Normalized value between 0 - 100")
    plt.xlabel("Date")
    plt.xticks(x, map(lambda y: y[5:], x_labels), rotation=60)
    ax.xaxis.set_major_locator(MultipleLocator(7))
    plt.legend(['# norms enabled', 'stringency index'])
    plt.title("# norms vs stringency index")
    plt.tight_layout()
    plt.show()


def find_all_events(values, print_all_dates=True):
    last_stringency = 0
    last_n_norms = 0

    print("Date\t\t|\tStringency\t|\t#norms\t")
    print("---------------------------------------")
    for date in sorted(list(values.keys())):
        if date > '2020-06-28':
            break
        stringency_changed = values[date]['stringency_index'] != last_stringency
        norms_changed = len(values[date]['norms']) != last_n_norms
        color = None
        if stringency_changed:
            last_stringency = values[date]['stringency_index']
            color = 'yellow'
        if norms_changed:
            last_n_norms = len(values[date]['norms'])
            color = 'magenta'
        if stringency_changed and norms_changed:
            color = 'green'
        if date < '2020-03-01' and last_stringency <= 0:
            continue
        if print_all_dates or stringency_changed or norms_changed:
            print(colored(
                '{0}\t|\t{1}\t\t|\t{2}\t'.format(
                    date,
                    last_stringency if stringency_changed else '\t',
                    last_n_norms if norms_changed else '\t'
                ),
                color)
            )


if __name__ == "__main__":
    # Download the US-timeseries_all XLSX file
    # all_virginia_values = load_virginia_stringency_data(sys.argv[1] if len(sys.argv) > 1 else None)
    # plot_number_of_norms_vs_stringency(all_virginia_values)
    # find_all_events(all_virginia_values, False)
    # load_more_data()
    # verify_metrics_for_stringency()
    get_dates_and_notes_for_stringency()