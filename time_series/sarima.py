# Goal:
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import logging
import sys

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class TimeSeriesModel:

    def __init__(self):
        self.data = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\BTC-USD_stock_data.csv')
        self.data = self.data[['Date', 'Close']]
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)

        self.train, self.test = [None] * 2

    def divide_train_test(self):
        logger.info("Divide train and test")
        proportion = 0.8
        line = int(len(self.data) * proportion)

        self.train = self.data.iloc[:line]
        self.test = self.data.iloc[line:]

    def differentiation_test(self):
        logger.info('Differentiation Assumption')

    def linearity(self):
        pass

    def resid_correlation(self):
        pass

    def normality_resid(self):
        pass

"""
ARIMA:
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class SimpleARIMA:
    def __init__(self, data, target_col, horizon):
        self.data = data.copy()
        self.target = target_col
        self.horizon = horizon

        self._prepare_data()

    def _prepare_data(self):
        self.data.index = pd.to_datetime(self.data.index) + pd.offsets.MonthEnd(0)
        self.data = self.data.asfreq('ME')
        self.series = self.data[self.target]

    def check_stationarity(self):
        p_value = adfuller(self.series)[1]
        return p_value

    def fit(self):
        print("ADF p-value:", self.check_stationarity())

        self.model = sm.tsa.ARIMA(
            self.series,
            order=(1, 1, 1),
            trend='t',
            enforce_stationarity=True,
            enforce_invertibility=True
        ).fit()

    def forecast(self):
        forecast = self.model.forecast(steps=self.horizon)

        future_dates = pd.date_range(
            start=self.series.index[-1] + pd.DateOffset(months=1),
            periods=self.horizon,
            freq='ME'
        )

        self.forecast_df = pd.DataFrame(
            {self.target: forecast.values},
            index=future_dates
        )

        # imprimir valores formatados
        print("\nPrevisão futura:")
        for date, value in self.forecast_df[self.target].items():
            print(f"{date.strftime('%Y-%m')} → {self.format_currency(value)}")

        return self.forecast_df

    def plot(self):
        def format_value(x, _):
            if abs(x) >= 1e6:
                return f'{x / 1e6:.1f}M'
            elif abs(x) >= 1e3:
                return f'{x / 1e3:.1f}K'
            return f'{x:.0f}'

        plt.figure(figsize=(10, 5))
        plt.plot(self.series, label='Histórico')
        plt.plot(self.forecast_df, label='Previsão', linestyle='--')
        plt.title(f'ARIMA Forecast - {self.target}')
        plt.xlabel('Data')
        plt.ylabel(self.target)
        plt.legend()

        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_value))
        plt.tight_layout()
        plt.show()

        print('=== DADOS HISTÓRICOS ===')
        for data, valor in self.series.items():
            print(f'{data:%Y-%m-%d}: {self.format_currency(valor)}')

    def format_currency(self, x):
        return f'{x:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
"""


class_time_series = TimeSeriesModel()
