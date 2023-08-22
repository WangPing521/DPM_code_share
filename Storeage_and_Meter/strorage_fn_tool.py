import pandas as pd
from typing import Dict, Any

def rename_df_columns(dataframe: pd.DataFrame, name: str, sep="_"):
    dataframe.columns = list(map(lambda x: name + sep + x, dataframe.columns))
    return dataframe

def OrderedDict2DataFrame(dictionary: Dict[int, Dict]):
    try:
        validated_table = pd.DataFrame(dictionary).T
    except ValueError:
        validated_table = pd.DataFrame(dictionary, index=[""]).T
    return validated_table
