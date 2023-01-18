import logging
from os.path import exists

import pandas as pd


class ResultsManager():
    """
        Singleton class to manage results.
    """
    _instance = None
    log = logging.getLogger('MAIN.RESULTS')
    multi_run_res = {}

    def __new__(cls, _=None):
        if cls._instance is None:
            cls._instance = super(ResultsManager, cls).__new__(cls)
        return cls._instance


    def __init__(self, metric='mAP@50'):
        if hasattr(self, 'results'):
            return
        columns = ['method', 'task', 'value', 'scenario']
        self.results = pd.DataFrame(columns=columns)
        self.metric = metric


    def has_results(self):
        return not self.results.empty


    def save_to_file(self, file_name=None):
        if not file_name:
            path = 'results/raw_results_df.pkl'
        else:
            path = 'results/' + file_name
        self.results.to_pickle(path)


    def load_from_file(self, file_name=None):
        if not file_name:
            path = 'results/raw_results_df.pkl'
        else:
            path = 'results/' + file_name
        if not exists(path):
            raise Exception('Results file not found')
        self.results = pd.read_pickle(path)


    def add_result(self, method, task, value, scenario):
        entry = pd.DataFrame([{
            'method' : method,
            'task': task,
            'value': value,
            'scenario': scenario
        }])
        self.results = pd.concat([self.results, entry], ignore_index=True)

        if method not in self.multi_run_res:
            self.multi_run_res[method] = {}
        if scenario not in self.multi_run_res[method]:
            self.multi_run_res[method][scenario] = {}
        if task not in self.multi_run_res[method][scenario]:
            self.multi_run_res[method][scenario][task] = []

        self.multi_run_res[method][scenario][task].append(value)


    def print_multiple_runs_results(self):
        if not self.multi_run_res:
            return

        from statistics import mean, variance, stdev

        self.log.info('------------ Multi run results ------------')
        for method, v2 in self.multi_run_res.items():
            self.log.info(f'\nMethod: {method}')
            for scenario, v1 in v2.items():
                self.log.info(f'\t\tScenario: {scenario}')
                for task, v in v1.items():
                    self.log.info(f'\t\tTask: {task}')#, v content: {v}')
                    self.log.info(f'\t\tMEAN: {mean(v):.3f}, VAR: {variance(v):.3f}, STDEV {stdev(v):.3f}')
        self.log.info('-------------------------------------------')


    def reset_results(self):
        if hasattr(self, 'summary'):
            delattr(self, 'summary')
        columns = ['method', 'task', 'value', 'scenario']
        self.results = pd.DataFrame(columns=columns)


    def generate_summary(self):
        self.summary = {}
        tasks = self.results.task.unique()
        methods = self.results.method.unique()
        self.summary['online'] = pd.DataFrame(columns=tasks)
        self.summary['offline'] = pd.DataFrame(columns=tasks)

        for method in methods:
            for scenario in ['online', 'offline']:
                df = self.results[(self.results['method'] == method) &
                                  self.results['scenario'].isin([scenario, None])]
                if not len(df):
                    continue
                self.summary[scenario].loc[method] = list(df['value'])


    def print_summary(self):
        if not hasattr(self, 'summary'):
            self.generate_summary()
        self.log.info('Results summary:')
        pd.set_option('display.max_columns', None)
        for scenario, scenario_summary in self.summary.items():
            self.log.info(scenario.upper(), ':')
            self.log.info(scenario_summary, '\n')

    def print_summary_latex(self, max_cols=8):
        self.log.info(f'\n{self.results}')
        import warnings
        from math import ceil
        warnings.simplefilter(action='ignore', category=FutureWarning)

        if not hasattr(self, 'summary'):
            self.generate_summary()

        res = ('-' * 30) + 'START LATEX' +  ('-' * 30)
        for scenario in self.summary.keys():
            hdrs = self.summary[scenario].columns.values
            # short_hdrs = [x.split('_')[0] for x in hdrs]
            short_hdrs = [x for x in hdrs]
            length = len(hdrs)
            if max_cols == 0 or max_cols > length:
                max_cols = length
            start = 0
            end = min(max_cols, length)
            num_splits = ceil(length / max_cols)
            res += "\n\\begin{table}\n\\centering\n\\caption{" + scenario.capitalize() + "}\n"
            for x in range(num_splits):
                res += self.summary[scenario].to_latex(float_format="%.1f",
                                                       columns=hdrs[start:end],
                                                       header=short_hdrs[start:end])
                if x < num_splits - 1:
                    res += "\\vspace{-.6mm}\\\\\n"

                start += max_cols
                if x == num_splits-2:
                    end = length
                else:
                    end += max_cols

            res += "\\end{table}\n"

        res += ('-' * 30) + 'END LATEX' +  ('-' * 30)
        self.log.info(res)


    def plot_summary(self, file_name=None):
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import seaborn as sns

        sns.set_style("whitegrid")
        g = sns.FacetGrid(data=self.results, col='scenario', hue='method',
                          legend_out=True, height=4, aspect= 1.33)
        g.map(sns.lineplot, 'task', 'value', marker='o')
        g.add_legend()

        for axes in g.axes.flat:
            ticks_loc = axes.get_xticks()
            axes.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

            # shorten x axis labels by cutting anything after an underscore
            # tasks_short = [x.get_text().split('_')[0] for x in axes.get_xticklabels()]
            # axes.set_xticklabels(tasks_short)

            axes.tick_params(labelleft=True)
            axes.set_xlabel('Task')
            axes.set_ylabel(self.metric)

        path = f'results/{file_name}' if file_name else 'results/plot_results.png'
        g.tight_layout()
        plt.savefig(path)
        # plt.show(block=True)


