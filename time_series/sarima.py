# Goal:
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
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
        p_value = adfuller(self.train)[1]

        if p_value > 0.05:
            self.train.loc[:, 'Close'] = self.train['Close'].diff()
            self.train = self.train.dropna().copy()

            plt.figure(figsize=(10, 5))
            plt.plot(self.train)
            plt.title('Data')
            plt.xlabel('Date')
            plt.ylabel('Diff data')
            plt.show()

    def find_parameters(self):
        logger.info('Parameter of ARIMA and Seasonal SARIMA')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        plot_acf(self.train, lags=10, ax=ax1)
        plot_pacf(self.train, lags=10, ax=ax2)

        plt.tight_layout()
        plt.show()

    def decomposition_data(self):
        pass

    def sarima_model(self):
        modelo = SARIMAX(self.train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        resultado = modelo.fit(disp=False)


    def resid_correlation(self):
        pass

    def normality_resid(self):
        pass

    def linearity_resid(self):
        pass

"""
ARIMA:
import numpy as np
import pandas as pd

import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker





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

import matplotlib.pyplot as plt
import statsmodels.api as sm

# Supondo que 'modelo_fit' é o seu modelo ARIMA treinado
residuos = modelo_fit.resid

# 1. Gráficos de diagnóstico automático do statsmodels
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
sm.graphics.tsa.plot_acf(residuos, lags=20, ax=ax[0, 0])
sm.graphics.tsa.plot_pacf(residuos, lags=20, ax=ax[0, 1])
ax[1, 0].plot(residuos)
ax[1, 0].set_title("Resíduos do Modelo")
sm.qqplot(residuos, line='s', ax=ax[1, 1])
ax[1, 1].set_title("QQ-plot dos Resíduos")

plt.tight_layout()
plt.show()

from statsmodels.stats.diagnostic import acorr_ljungbox

# Realiza o teste de Ljung-Box com as 10 primeiras defasagens
resultado_ljungbox = acorr_ljungbox(residuos, lags=[10], return_df=True)
print(resultado_ljungbox)

"""


class_time_series = TimeSeriesModel()
class_time_series.divide_train_test()
class_time_series.differentiation_test()
class_time_series.find_parameters()