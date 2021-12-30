"""
A file containing code experiments to find events in the base files

See https://github.com/OxCGRT/USA-covid-policy
and specifically https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md#containment-and-closure-policies

Visualized at https://ourworldindata.org/grapher/covid-stringency-index?tab=chart&time=2020-01-22..2020-04-25&country=~USA

Differences exist between the Oxford data set and the the norm schedule derived for the studies encoded in this repository,
which has been derived from https://coronavirus.jhu.edu/data/state-timeline/new-confirmed-cases/virginia/3
"""
from collections import defaultdict
from datetime import datetime

from termcolor import colored

from OxCGRTUS_utils import *
from load_us_data import split_norm_line


def compare_rows(prev, next, cat, flag, notes):
    if prev is None:
        return True
    if prev[cat] != next[cat] and not (pd.isna(prev[cat]) and pd.isna(next[cat])):
        return False
    elif flag in prev and prev[flag] != next[flag] and not (pd.isna(prev[flag]) and pd.isna(next[flag])):
        return False
    # elif prev[notes] != next[notes] and not pd.isna(next[notes]):
    #     return False
    else:
        return True


def get_events_for_flag(df, category_name):
    cat = next(filter(lambda x: x.startswith(category_name) and not (x.endswith('_Flag') or x.endswith('_Notes')), df.columns))
    flag = f"{category_name}_Flag"
    notes = f"{category_name}_Notes"
    last_row = None
    events = dict()
    for row in df.to_dict(orient="records"):
        if row['Date'] <= 20200628:
            if not compare_rows(last_row, row, cat, flag, notes):
                date = datetime.strftime(datetime.strptime(str(row['Date']), "%Y%m%d"), "%Y-%m-%d")
                events[date] = dict(date=date, category=category_name, id=cat, measurement=row[cat], notes=row[notes], stringency=row['StringencyIndex'])
                if flag in row:
                    events[date][flag] = row[flag]
        last_row = row

    return cat, events


def get_dates_and_notes_for_stringency():
    filename = 'OxCGRT_US_latest.csv'
    save_excel_file_from_github(filename=filename)
    df = pd.read_csv(filename)
    virginia_df = df[df['RegionCode'] == 'US_VA']

    all_events = defaultdict(dict)

    # indicators = dict(C=[1, 2, 3, 4, 5, 6, 7, 8], E=[1, 2, 3, 4], H=[1, 2, 3, 4, 5, 6, 7, 8])
    indicators = dict(C=[1, 2, 3, 4, 5, 6, 7, 8], H=[1])
    for indicator_category in indicators:
        for subcategory in indicators[indicator_category]:
            name, events = get_events_for_flag(virginia_df, f"{indicator_category}{subcategory}")
            for event, values in events.items():
                all_events[event][name] = values

    with open('norm-events.csv', 'w') as norm_events_out:
        norm_events_out.write("date;stringency;category;value;norm;param;notes\n")
        for date in sorted(all_events):
            for category, event in all_events[date].items():
                norm_events_out.write(f"{date};{event['stringency']};{event['category']};{event['measurement']};;;\"{event['notes']}\"\n")

    for date, events in load_existing_norm_schedule_events().items():
        all_events[date]['norm-schedule'] = events

    dates = sorted(all_events)
    stringency = 0
    for date in dates:
        events = all_events[date]
        next_stringency = list(filter(lambda x: "stringency" in x, events.values()))
        if next_stringency:
            stringency = next_stringency[0]['stringency']
        print(date, colored(stringency, 'yellow'))
        for event in events:
            if event == 'norm-schedule':
                if events[event]['start']:
                    for norm in events[event]['start']:
                        print("\t", colored(
                            f"{norm['norm']}[{norm['params']}] -- {norm['category']}: {norm['category_value']}", 'green'))
                if events[event]['end']:
                    for norm in events[event]['end']:
                        print("\t", colored(f"{norm['norm']}[{norm['params']}] -- {norm['category']}: {norm['category_value']}", 'red'))
            else:
                print("\t", colored(f"{event}: {events[event]['measurement']}", 'blue'), events[event]['notes'])


def load_existing_norm_schedule_events():
    events = defaultdict(lambda: defaultdict(list))
    with open('norm-schedule.csv', 'r') as norms_in:
        headers = norms_in.readline()[:-1]
        for line in norms_in:
            line = split_norm_line(line[:-1])
            start = line[0]
            end = line[5]
            values = dict(norm=line[1], params=line[6], category=line[2], category_value=line[3], start=start, end=end)
            if values['norm'] != '':
                events[start]['start'].append(values)
                if end != '':
                    events[end]['end'].append(values)

    return events


if __name__ == "__main__":
    get_dates_and_notes_for_stringency()
