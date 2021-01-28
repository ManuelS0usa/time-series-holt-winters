import matplotlib.pyplot as plt
import numpy as np

# Time Series
# from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class Algoritmos_de_Series_Temporais:

    def holt_winters(self, data, dias_previsao, sazonalidade_diaria, sazonalidade_horaria):
        data = np.array(data)

        model_holt_winters = ExponentialSmoothing(
            data,
            seasonal_periods=sazonalidade_horaria * sazonalidade_diaria,
            seasonal='add'
        )
        fit_holt_winter = model_holt_winters.fit()
        # previsão
        forecast_values = fit_holt_winter.forecast(dias_previsao)

        return forecast_values

    def holt_winters_testing_prediction(self, data, dias_previsao, sazonalidade_diaria, sazonalidade_horaria):
        data = np.array(data)
        treino = data[:len(data) - (7)]
        teste = data[len(treino):]

        model_holt_winters = ExponentialSmoothing(
            treino,
            seasonal_periods=sazonalidade_horaria * sazonalidade_diaria,
            seasonal='add'
        )

        fit_holt_winter = model_holt_winters.fit()
        # previsão
        # forecast_values = fit_holt_winter.forecast(
        # dias_previsao*sazonalidade_diaria)
        model_values = fit_holt_winter.fittedvalues

        # fitted values
        plt.plot(treino, label='original')
        plt.plot(model_values, label='Holt-winters')
        plt.legend()
        plt.show()

        # previsão
        prediction_size = len(teste)
        forecast_values = fit_holt_winter.forecast(prediction_size)
        plt.plot(teste, label='original')
        plt.plot(forecast_values, label='forecast')
        plt.legend()
        plt.show()

        pred1 = [0] * len(treino)
        for i in forecast_values:
            pred1 = np.append(pred1, i)

        plt.plot(data, label='original')
        plt.plot(pred1, label='forecast')
        plt.legend()
        plt.show()

        return forecast_values
