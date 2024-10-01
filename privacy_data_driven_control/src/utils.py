
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Any

def processing_results(loaded_data: Dict[str, Any],
                        bins: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:

   
    disturbance_gamma_results = loaded_data["disturbance_gamma_results"]
    max_of_disturbance = loaded_data["max_of_disturbance"]
    num_data_sets = loaded_data["num_data_sets"]
    # print(disturbance_gamma_results)
    df = pd.DataFrame(data=disturbance_gamma_results, 
                    columns=max_of_disturbance)


    cut_gamma_bar = []

    for i in range(np.size(disturbance_gamma_results, 1)):
        z = pd.cut(disturbance_gamma_results[:,i], bins=bins)
        y = pd.get_dummies(z, dtype=float).sum(axis=0)
        cut_gamma_bar.append(y)


    df_gamma_bar_dmax = pd.DataFrame(cut_gamma_bar).transpose()
    df_gamma_bar_dmax = df_gamma_bar_dmax/num_data_sets
    df_gamma_bar_dmax = df_gamma_bar_dmax.loc[::-1]
    df_gamma_bar_dmax.columns = max_of_disturbance


    return df, df_gamma_bar_dmax

