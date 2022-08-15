"""
This file contains an effort to create really strong plots about mobility and calibration
"""
import datetime
import os
import re
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


@click.command()
@click.option(
    "-m", "--mobility-index-csv",
    help="Location of the CSV file containing the Quebic mobility indices for each county",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
    required=True
)
@click.option(
    "-n", "--norm-schedule",
    help="Location of the norm schedule CSV file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
    required=False
)
@click.option(
    "-s", "--simulation-output",
    help="A directory containing one directory for each simulation run, with the simulation output in them",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    required=False
)
@click.option(
    "-a/-no-a", "--average/--no-average",
    help="If this argument is passed, the average of all (passed) counties will be plotted. Otherwise, each county will "
         "get a separate line",
    required=False,
    default=True,
    type=bool
)
@click.option(
    "-t", "--title",
    help="Title of the plot. Also used to save this figure",
    type=str,
    required=False
)
@click.option(
    "-d", "--day-averages",
    help="To calculate the mobility index, the baseline average mobility for each day of the week needs to be known."
         " If this location is not present, real mobility will be plotted for simulation output instead of index",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    required=False
)
@click.argument(
    "fips_codes",
    nargs=-1
)
def make_plot(**args):
    BetterMobilityPlotter(**args)


class BetterMobilityPlotter(object):

    def __init__(self, mobility_index_csv, norm_schedule, fips_codes, simulation_output, average, day_averages, title):
        df = self.load_mobility(mobility_index_csv)
        if fips_codes:
            df = df[['date'] + list(map(str, fips_codes))]
        self.norm_schedule = norm_schedule
        self.simulation_output = simulation_output
        self.day_averages = day_averages

        func = self.plot_df_average if average else None
        self.__plot(func, df, title)

    @staticmethod
    def load_mobility(path):
        df = pd.read_csv(path)
        dates = list(df['date'])
        min_date = datetime.datetime.strptime(df[['date']].min()[0], "%Y-%m-%d")
        max_date = datetime.datetime.strptime(df[['date']].max()[0], "%Y-%m-%d")
        while min_date < max_date:
            min_date += datetime.timedelta(days=1)
            ds = min_date.strftime("%Y-%m-%d")
            if ds not in dates:
                missing_row = pd.DataFrame(
                    [[ds] + [pd.NA for _ in range(len(df.columns) - 1)]],
                    columns=list(df.columns),
                    index=[df.shape[0]]
                )
                df = df.append(missing_row)
        return df.sort_values('date').rolling(7, on='date', min_periods=1).mean()

    @staticmethod
    def plot_df_average(ax, x_labels, df):
        line = ax.plot(x_labels, df.mean(axis=1), label="Cuebiq Mobility Data")
        ax.fill_between(
            x_labels,
            df.mean(axis=1) - df.std(axis=1),
            df.mean(axis=1) + df.std(axis=1),
            alpha=0.2,
            facecolor=line[0].get_color()
        )

    @staticmethod
    def plot_df_counties(ax, x_labels, df):
        # TODO
        pass

    def __plot(self, ax_func, df, title=None):
        fig, ax = plt.subplots()
        x_labels = df['date']

        dates = [x for x in sorted(list(x_labels)) if "2020-03-01" <= x <= "2020-06-29"]
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

        x_ticks = [x for x in dates if x.endswith("01") or x.endswith("15") or x == "2020-06-28"]
        x_tick_labels = [x[len("2020-01-"):] if x.endswith("15") or x.endswith("28") else months[
            int(x[len("2020-"):len("2020-") + 2]) - 1] for x in x_ticks]

        ax_func(ax, x_labels, df)
        self.__add_simulation_averages(ax, x_labels)
        self.__add_norm_schedule(ax)

        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_xlim([7, len(x_labels)])
        ax.set_ylabel("Mobility Index (pct change w.r.t. pre-covid period)", fontdict=dict(size=8))
        ax.set_xlabel("Date in 2020")
        ax.yaxis.set_major_formatter(PercentFormatter())

        plt.legend()

        if title:
            plt.title(title)
            fig.set_dpi(300)
            fig.set_size_inches(6.8, 3.4)
            plt.savefig(title + ".png", dpi=300)
        plt.tight_layout()
        plt.show()

    def __add_norm_schedule(self, ax):
        if not self.norm_schedule:
            return
        starts, ends = self.__load_norm_events()
        for event in sorted(list(set(starts + ends))):
            color = 'red' if event in starts and event not in ends else 'green' if event not in starts else 'orange'
            ax.axvline(event, color=color, alpha=0.3)

    def __load_norm_events(self):
        starts = list()
        ends = list()
        with open(self.norm_schedule, 'r') as norms_in:
            headers = norms_in.readline()[:-1].split(",")
            for line in norms_in:
                norm_event = dict(zip(headers, line[:-1].split(",")))
                starts.append(norm_event['start'])
                if norm_event['end']:
                    ends.append(norm_event['end'])
        return sorted(starts), sorted(ends)

    def __add_simulation_averages(self, ax, x_labels):
        if not self.simulation_output:
            return
        avg, std = self.__load_simulation_averages(x_labels)
        line = ax.plot(x_labels, avg, label="Simulated")
        ax.fill_between(x_labels, avg - std, avg + std, facecolor=line[0].get_color(), alpha=0.2)

    def __load_simulation_averages(self, x_labels):
        avg, std = dict(), dict()
        simulation_data = self.__load_simulation()
        for date, counties in simulation_data.items():
            date_vals = list()
            for county, vals in counties.items():
                date_vals += vals
            avg[date] = np.mean(date_vals)
            std[date] = np.std(date_vals)

        return (
            pd.Series([avg[x] for x in x_labels]).rolling(7).mean(),
            pd.Series([std[x] for x in x_labels]).rolling(7).mean()
        )

    def __add_simulation_per_county(self, ax):
        # TODO
        pass

    def __load_simulation(self):
        day_avg_dct = self.__load_day_averages()

        simulation_outputs = [
            os.path.join(self.simulation_output, x, 'tick-averages.csv')
            for x in os.listdir(self.simulation_output)
            if 'tick-averages.csv' in os.listdir(os.path.join(self.simulation_output, x))
        ]
        data = defaultdict(lambda: defaultdict(list))
        for ta in simulation_outputs:
            with open(ta, 'r') as tain:
                for line in tain:
                    date, fips, radius, n_agents = line[:-1].split(",")
                    index = self.__get_mobility_index(day_avg_dct, date, fips, radius)
                    data[date][int(fips)].append(float(index))

        return data

    @staticmethod
    def __get_mobility_index(day_averages, date, county_fips, radius):
        dt = datetime.datetime.strptime(date, "%Y-%m-%d")
        baseline = float(day_averages[int(county_fips)][dt.weekday()])
        return (float(radius) - baseline) / baseline * 100

    def __load_day_averages(self):
        day_averages_dct = dict()
        if self.day_averages:
            for f_day_average in os.listdir(self.day_averages):
                match = re.findall(r'[a-zA-Z-]+(\d+)[a-zA-Z-]*.csv', f_day_average)
                if match:
                    fips = int(match[0])
                    day_averages_dct[fips] = dict()
                    with open(os.path.join(self.day_averages, f_day_average), 'r') as day_av_in:
                        for line in day_av_in:
                            day, avg = line.split(",")
                            day_averages_dct[fips][int(day)] = float(avg)
        return day_averages_dct


if __name__ == "__main__":
    make_plot()
