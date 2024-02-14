import numpy as np
from pandas import DataFrame


def get_porcentaje_faltantes(dataframe: DataFrame) -> float:
    return np.abs((dataframe.count() - dataframe.shape[0]) / dataframe.shape[0] * 100)



