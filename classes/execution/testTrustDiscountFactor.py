from typing import List, Dict

from classes.execution.CodeExecution import CodeExecution


class TestTrustDiscountFactor(CodeExecution):
    rundirectory_template = [
        "trust-discount",
        "{ncounties}counties-fips-{fips}",
        "trust-discount-factor-{discount_factor}-sample-random-{random_sample}-mode-liberal-{mode_liberal}-mode-conservative-{mode_conservative}-run{run}",
    ]

    progress_format = "[TRUST DISCOUNT]: Discount factor {discount_factor} using random sampling = {random_sample} with modes (lib, cons) of ({mode_liberal}, {mode_conservative})"
    target_file = "tick-averages.csv"

    def __init__(self, *args, **kwargs):
        super(TestTrustDiscountFactor, self).__init__(*args, **kwargs)
        self.discount_factor = 0
        self.random_sample = False
        self.saved_liberal = kwargs['mode_liberal']
        self.saved_conservate = kwargs['mode_conservative']
        # self.initiate()
        self.run_discount_simulations(1, 1, False)

    def initiate(self):
        self.run_discount_simulations(0, 0, True)
        for lib, cons in [(0.5, 0.5), (1, 1), (self.saved_liberal, self.saved_conservate)]:
            self.run_discount_simulations(lib, cons, False)

    def run_discount_simulations(self, mode_liberal, mode_conservative, sample_random):
        self.run_configuration['random_sample'] = sample_random
        self.mode_liberal = mode_liberal
        self.mode_conservative = mode_conservative
        self.run_configuration['mode_liberal'] = mode_liberal
        self.run_configuration['mode_conservative'] = mode_conservative
        for i in range(11):
            self.discount_factor = i / 10
            self.run_configuration['discount_factor'] = self.discount_factor
            self.calibrate(None)

    def store_fitness_guess(self, x):
        pass

    def prepare_simulation_run(self, x):
        pass

    def score_simulation_run(self, x: tuple, directories: List[Dict[int, str]]) -> float:
        pass

    def _write_csv_log(self, score):
        pass

    def get_extra_java_commands(self):
        args = [
            "--trust-discount",
            str(self.discount_factor)
        ]
        if self.random_sample:
            args.append("--sample-random")

        return args

    def _process_loss(self, _, __):
        pass
