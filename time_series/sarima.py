# Goal: Predict climate time series with SARIMA
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.gofplots import qqplot
from sklearn.metrics import mean_squared_error, root_mean_squared_error
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
        self.train_data = pd.read_csv('files\\DailyDelhiClimateTrain.csv')
        self.test_data = pd.read_csv('files\\DailyDelhiClimateTest.csv')

        self.train_data['date'] = pd.to_datetime(self.train_data['date'])
        self.test_data['date'] = pd.to_datetime(self.test_data['date'])

        self.train_data.set_index('date', inplace=True)
        self.test_data.set_index('date', inplace=True)

        self.train_data = self.train_data.resample('W').mean()
        self.test_data = self.test_data.resample('W').mean()

        self.train_data = self.train_data.asfreq('W')
        self.test_data = self.test_data.asfreq('W')

        self.X_train = self.train_data[['humidity']]
        self.y_train = self.train_data[['meantemp']]

        self.y_test = self.test_data[['meantemp']]
        self.X_test = self.test_data[['humidity']]

        self.d_order, self.model, self.forecast = 0, None, None

    def decomposition_data(self):
        logger.info('Decomposition time series')

        decomposition = seasonal_decompose(self.y_train["meantemp"], model="additive", period=7)

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
            self.y_train.loc[:, 'meantemp'] = self.y_train['meantemp'].diff()
            self.y_train = self.y_train.dropna().copy()
            self.d_order += 1
            p_value = adfuller(self.y_train["meantemp"])[1]

        logger.debug(f'Data Series is stationary. P_value: {p_value:.4f}')
        plt.figure(figsize=(10, 5))
        plt.plot(self.y_train)
        plt.title('Data')
        plt.xlabel('Date')
        plt.ylabel('Diff data')
        plt.show()

    def find_parameters(self):
        logger.info('Parameter of ARIMA and Seasonal SARIMA')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        plot_acf(self.y_train['meantemp'], lags=30, ax=ax1)
        plot_pacf(self.y_train['meantemp'], lags=30, ax=ax2)

        plt.tight_layout()
        plt.show()

    def sarima_model(self):
        logger.info('Fitting model')
        model = SARIMAX(self.y_train, exog=self.X_train, order=(1, self.d_order, 0), seasonal_order=(1, 0, 0, 52))
        self.model = model.fit(disp=False)
        logger.info(self.model.summary())

    def resid_correlation(self):
        logger.info("Visualizing Residual correlation")
        resid = self.model.resid

        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(resid, lags=30, ax=ax)
        plt.title("ACF of resid (Everything must stay within the blue zone.)")
        plt.show()

    def normality_resid(self):
        logger.info("Visualizing Residual Normality - Points should follow the red diagonal line")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        self.model.resid.hist(ax=ax1, bins=20, edgecolor="black")
        ax1.set_title("Hist of resid")

        # Gráfico Q-Q ()
        qqplot(self.model.resid, line="s", ax=ax2)
        ax2.set_title("Gráfico Q-Q Normal")
        plt.show()

    def linearity_resid(self):
        logger.info("Visualizing Residual Linearity - The dots should be scattered randomly without forming "
                    "patterns or curves.")

        plt.figure(figsize=(10, 4))
        plt.scatter(self.model.fittedvalues, self.model.resid, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.title("Linearity of adjust values X resid")
        plt.xlabel("Adjust values")
        plt.ylabel("Resid")
        plt.show()

    def plot_predictions(self):
        logger.info("Generating and plotting forecasts")

        forecast_object = self.model.get_forecast(
            steps=len(self.X_test), exog=self.X_test
        )
        self.forecast = forecast_object.predicted_mean
        confidence_intervals = forecast_object.conf_int()

        plt.figure(figsize=(12, 6))
        plt.plot(
            self.y_train.index[-52:],
            self.y_train.values[-52:],
            label="Train (Last year)",
            color="gray",
            alpha=0.6,
        )
        plt.plot(
            self.y_test.index,
            self.y_test.values,
            label="Real Data (Test)",
            color="black",
            linewidth=2,
        )
        plt.plot(
            self.y_test.index,
            self.forecast,
            label="Predict (With Exog)",
            color="blue",
            linestyle="--",
            linewidth=2,
        )
        plt.fill_between(
            self.y_test.index,
            confidence_intervals.iloc[:, 0],  # Inferior limit
            confidence_intervals.iloc[:, 1],  # Superior limit
            color="blue",
            alpha=0.15,
            label="Confidence interval (95%)",
        )
        plt.title("Comparison: Actual Average Temperature vs. SARIMA Forecast")
        plt.xlabel("Data")
        plt.ylabel("Temperatura Média")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def evaluating_model(self):
        logger.info("Looking if our model is good to use")
        mse = mean_squared_error(self.y_test['meantemp'], self.forecast)
        root_mean = root_mean_squared_error(self.y_test['meantemp'], self.forecast)

        logger.info(f'Mean Squared Error: {mse}')
        logger.info(f'Root Mean Squared Error: {root_mean}')


class_time_series = TimeSeriesModel()
class_time_series.decomposition_data()
class_time_series.differentiation_test()
class_time_series.find_parameters()
class_time_series.sarima_model()
class_time_series.resid_correlation()
class_time_series.normality_resid()
class_time_series.linearity_resid()
class_time_series.plot_predictions()
class_time_series.evaluating_model()