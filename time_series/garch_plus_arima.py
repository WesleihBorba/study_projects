# Goal: Predict stocks using Arima and Garch together - https://www.kaggle.com/datasets/camnugent/sandp500
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
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


class GarchArimaModel:

    def __init__(self):
        self.data = pd.read_csv('files\\AMD_data.csv')

        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)
        self.data = self.data.asfreq('B')
        self.data = self.data.ffill()
        self.data = self.data[['close']]

        self.garch, self.garch_resid = [None] * 2
        self.arima, self.arima_resid = [None] * 2
        self.train, self.test = [None] * 2
        self.q_garch, self.p_garch = [None] * 2
        self.q_arima, self.p_arima = [None] * 2

    def adjust_return(self):
        logger.info('Adjusting return for finance')
        self.data['log_return'] = np.log(self.data['close'] / self.data['close'].shift(1)) * 100
        self.data = self.data.dropna()
        self.data.drop(columns={'close'}, inplace=True)

    def train_test(self):
        logger.info('Dividing in train and test')
        split_point = int(len(self.data) * 0.8)
        self.train = self.data.iloc[:split_point]
        self.test = self.data.iloc[split_point:]

    def find_parameters(self):
        logger.info('Parameter of ARIMA')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        plot_acf(self.train['log_return'], lags=30, ax=ax1)
        plot_pacf(self.train['log_return'], lags=30, ax=ax2)

        plt.tight_layout()
        plt.show()

    def arima_model(self):
        logger.info('Running Arima')
        self.arima = ARIMA(self.train, order=(0, 0, 0)).fit()
        self.arima_resid = self.arima.resid
        logger.info(self.arima.summary())

    def resid_correlation(self):
        logger.info("Visualizing Residual correlation")

        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(self.arima_resid, lags=30, ax=ax)
        plt.title("ACF of resid (Everything must stay within the blue zone.)")
        plt.show()

    def linearity_resid(self):
        logger.info("Visualizing Residual Linearity - and running Ljung-Box test")

        ljung_box_test = acorr_ljungbox(self.arima_resid, lags=[10], return_df=True)
        logger.info(f'p-value < 0.05 proves the existence of an ARCH effect, then run Garch: {ljung_box_test}')

        plt.figure(figsize=(10, 4))
        plt.scatter(self.arima.fittedvalues, self.arima_resid, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.title("Linearity of adjust values X resid")
        plt.xlabel("Adjust values")
        plt.ylabel("Resid")
        plt.show()

    def hist_resid(self):
        logger.info('How parameter will use in "dist" using Garch model. If the ends of the histogram are higher '
                    'than the normal curve (heavy tails), choose dist="t" or dist="skewt"')
        plt.hist(self.arima_resid)
        plt.xlabel('Data')
        plt.ylabel('Return')
        plt.title('Hist of Arima Resid')
        plt.show()

    def garch_model(self):
        logger.info('Running Garch Model')
        self.garch = arch_model(self.arima_resid,
                                mean='Zero',  # GARCH model that the mean has already been cleaned by the ARIMA model
                                p=1,
                                q=1,
                                dist='t'  # Getting of hist_resid
                                ).fit()
        self.garch_resid = self.garch.resid / self.garch.conditional_volatility

        logger.info('Look p-values < 0.05 and alpha_1 + beta_1 < 1')
        logger.info(self.garch.summary())

        ljung_box_test = acorr_ljungbox(self.arima_resid, lags=[10], return_df=True)
        logger.info(f'p-value > 0.05 demonstrating that GARCH absorbed all the risk, leaving only pure white noise: '
                    f'{ljung_box_test}')

    def validation_model(self):
        logger.info("Running Dynamic Rolling Forecast for Validation")

        arima_predictions = []
        garch_volatility_predictions = []

        history_returns = self.train['log_return'].to_list()

        for i in range(len(self.test)):
            current_mean = np.mean(history_returns)
            arima_predictions.append(current_mean)

            current_residuals = np.array(history_returns) - current_mean

            temp_garch = arch_model(current_residuals, mean='Zero', p=1, q=1, dist='t')
            temp_garch_res = temp_garch.fit(disp='off', show_warning=False)

            garch_forecast = temp_garch_res.forecast(horizon=1)
            next_step_variance = garch_forecast.variance.iloc[-1].values[0]
            next_step_volatility = np.sqrt(next_step_variance)
            garch_volatility_predictions.append(next_step_volatility)

            actual_value = self.test['log_return'].iloc[i]
            history_returns.append(actual_value)

        arima_forecast_series = pd.Series(arima_predictions, index=self.test.index)
        volatility_series = pd.Series(garch_volatility_predictions, index=self.test.index)

        upper_superior = arima_forecast_series + (2 * volatility_series)
        lower_inferior = arima_forecast_series - (2 * volatility_series)

        inside_band = (self.test['log_return'] >= lower_inferior) & (self.test['log_return'] <= upper_superior)
        coverage_rate = (inside_band.sum() / len(self.test)) * 100
        logger.info(f"GARCH Band Coverage Rate: {coverage_rate:.2f}%")

        plt.figure(figsize=(14, 7))
        plt.plot(self.test.index, self.test['log_return'].values, label='Real Return of AMD (Teste)', color='gray',
                 alpha=0.5)
        plt.plot(arima_forecast_series.index, arima_forecast_series.values, label='Predict ARIMA (Mean)',
                 color='blue', lw=1.5, linestyle='--')

        plt.fill_between(self.test.index, lower_inferior, upper_superior, color='crimson', alpha=0.18,
                         label=f'Volatility of GARCH (95% confidence)')

        plt.title(f'Dynamic Validation ARIMA-GARCH (AMD) - Coverage: {coverage_rate:.2f}%')
        plt.xlabel('Data')
        plt.ylabel('Log-Return (%)')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


model_class = GarchArimaModel()
model_class.adjust_return()
model_class.train_test()
model_class.find_parameters()
model_class.arima_model()
model_class.resid_correlation()
model_class.linearity_resid()
model_class.hist_resid()
model_class.garch_model()
model_class.validation_model()