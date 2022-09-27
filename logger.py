import pandas as pd
import os


class Logger:
    def __init__(self, *keys):
        self.log = {key: [] for key in keys}

    def log_data(self, *data):
        for i, value in enumerate(self.log.values()):
            value.append(data[i])

    def file_data(self, directory):
        df = pd.DataFrame.from_dict(self.log)
        if not os.path.exists(f'./{directory}'):
            os.mkdir(f'./{directory}')
        df.to_csv(f'./{directory}/data.csv', index_label='time step')
