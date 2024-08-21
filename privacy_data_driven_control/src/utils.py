
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Any

def processing_results(loaded_data: Dict[str, Any],
                        bins: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:

   
    disturbance_epsilon_results = loaded_data["disturbance_epsilon_results"]
    max_of_disturbance = loaded_data["max_of_disturbance"]
    num_data_sets = loaded_data["num_data_sets"]
    # print(disturbance_epsilon_results)
    df = pd.DataFrame(data=disturbance_epsilon_results, 
                    columns=max_of_disturbance)


    cut_epsbar = []

    for i in range(np.size(disturbance_epsilon_results, 1)):
        z = pd.cut(disturbance_epsilon_results[:,i], bins=bins)
        y = pd.get_dummies(z, dtype=float).sum(axis=0)
        cut_epsbar.append(y)


    df_epsbar_dmax = pd.DataFrame(cut_epsbar).transpose()
    df_epsbar_dmax = df_epsbar_dmax/num_data_sets
    df_epsbar_dmax = df_epsbar_dmax.loc[::-1]
    df_epsbar_dmax.columns = max_of_disturbance


    return df, df_epsbar_dmax

