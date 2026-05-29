# Goal: Predict climate time series with SARIMA
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
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
        self.train_data = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\DailyDelhiClimateTrain.csv')
        self.test_data = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\DailyDelhiClimateTest.csv')

        self.train_data['date'] = pd.to_datetime(self.train_data['date'])
        self.test_data['date'] = pd.to_datetime(self.test_data['date'])

        self.train_data.set_index('date', inplace=True)
        self.test_data.set_index('date', inplace=True)

        self.train_data = self.train_data.asfreq('D')
        self.test_data = self.test_data.asfreq('D')

        self.X_train = self.train_data[['humidity', 'meanpressure', 'wind_speed']]
        self.y_train = self.train['meantemp']

        self.y_test = self.test_data['meantemp']
        self.X_test = self.test_data[['humidity', 'meanpressure', 'wind_speed']]

        self.d_order = 0

    def decomposition_data(self):
        logger.info('Decomposition time series')

        decomposition = seasonal_decompose(self.y_train["mean_temp"], model="additive", period=7)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        decomposition.trend.plot(ax=ax1, title="Trend")
        decomposition.seasonal.plot(ax=ax2, title="Seasonal")
        decomposition.resid.plot(ax=ax3, title="Resid")
        plt.tight_layout()
        plt.show()

    def differentiation_test(self):
        logger.info('Differentiation Assumption')
        p_value = adfuller(self.y_train)[1]

        while p_value > 0.05:
            logger.debug(f'Data not stationary. P_value {p_value:.4f}')
            self.train.loc[:, 'Close'] = self.train['Close'].diff()
            self.train = self.train.dropna().copy()
            self.d_order += 1
            p_value = adfuller(self.train["Close"])[1]

        logger.debug(f'Data Series is stationary. P_value: {p_value:.4f}')
        plt.figure(figsize=(10, 5))
        plt.plot(self.train)
        plt.title('Data')
        plt.xlabel('Date')
        plt.ylabel('Diff data')
        plt.show()

    def find_parameters(self):
        logger.info('Parameter of ARIMA and Seasonal SARIMA')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        plot_acf(self.train['Close'], lags=30, ax=ax1)
        plot_pacf(self.train['Close'], lags=30, ax=ax2)

        plt.tight_layout()
        plt.show()

    def sarima_model(self):
        logger.info('Fitting model')
        model = SARIMAX(self.train, order=(0, self.d_order, 0), seasonal_order=(0, 0, 0, 7))
        results = model.fit(disp=False)
        logger.info(results.summary())


    def resid_correlation(self):
        pass

    def normality_resid(self):
        pass

    def linearity_resid(self):
        pass

"""


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
#class_time_series.divide_train_test()
#class_time_series.decomposition_data()
#class_time_series.differentiation_test()
#class_time_series.find_parameters()