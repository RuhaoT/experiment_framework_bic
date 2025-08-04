"""This module provides logging utilities for the experiment framework.

"""

import types
import os
import time
import dataclasses
import json
import logging

import pandas as pd
import tomlkit
import dacite

from auto_utils import combiparam


def init_experiment_folder(
    data_folder: str, experiment_name: str = "data", timed: bool = True
) -> str:
    """This function initializes the experiment folder.

    Args:
        experiment_name (str): The name of the experiment.
        data_folder (str): The folder to store the experiment data.
    """
    # create data folder
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # create experiment folder
    if timed:
        experiment_folder_name = experiment_name + "_" + time.strftime("%Y%m%d-%H%M%S")
    else:
        experiment_folder_name = experiment_name
    experiment_folder = os.path.join(data_folder, experiment_folder_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    else:
        # raise an warning if the folder already exists
        print(f"Warning: {experiment_folder} already exists.")

    return experiment_folder

def save_dataclasses_to_csv(param_list: list[dataclasses.dataclass], path:str, filename: str="params.csv"):
    """Save the parameters to a csv file."""
    # convert the dataclass to a list of dictionaries
    param_dict_list = [dataclasses.asdict(param) for param in param_list]
    
    # normalize the dict to a dataframe
    param_df = pd.json_normalize(param_dict_list)
    
    # save the dataframe to csv
    full_path = os.path.join(path, filename)
    param_df.to_csv(full_path, index=False)
    return full_path

def save_dataclass_to_json(
    dataclass: dataclasses.dataclass, filepath: str, encoding: str = "utf-8"
):
    """Saves a dataclass to a json file.

    Args:
        dataclass (dataclass): The dataclass to save.
        filepath (str): The path to save the dataclass.
    """
    data = dataclasses.asdict(dataclass)
    data = serialize_combiparam_dict(data)
    with open(filepath, "w", encoding=encoding) as f:
        json.dump(data, f)
        
def save_dataclass_to_toml(
    dataclass: dataclasses.dataclass, filepath: str, encoding: str = "utf-8"
):
    """Saves a dataclass to a TOML file.

    Args:
        dataclass (dataclass): The dataclass to save.
        filepath (str): The path to save the dataclass.
    """
    data = dataclasses.asdict(dataclass)
    data_serialized = serialize_combiparam_dict(data)
        
    # iterate over the original data and add comments to each key
    def recursive_add_comment(current_toml_position, current_dict_position):
        """Recursively adds comments to the TOML structure."""
        for key, value in current_dict_position.items():
            if isinstance(value, dict):
                # If the value is a dictionary, recurse into it
                new_toml_position = current_toml_position[key]
                recursive_add_comment(new_toml_position, value)
            else:
                # Add a comment for the key indicating data type
                key_type = type(current_dict_position[key]).__name__
                comment = f"Type: {key_type}"
                if isinstance(value, combiparam.Combiparam):
                    # If the value is a Combiparam, add its type to the comment
                    comment += f", {value.val_info()}"
                current_toml_position[key].comment(comment)
                logging.debug(f"Added comment for {key}: {comment}")
    
    # dump the data to a TOML structure
    toml_data = tomlkit.dumps(data_serialized)
    toml_data = tomlkit.parse(toml_data)
    
    # Add comments to the TOML structure
    recursive_add_comment(toml_data, data)
    
    # write the TOML data to the file
    with open(filepath, "w", encoding=encoding) as f:
        tomlkit.dump(toml_data, f)

def load_dataclass_from_toml(filepath: str, dataclass_type: type) -> dataclasses.dataclass:
    """Loads a dataclass from a TOML file.

    Args:
        filepath (str): The path to the TOML file.
        dataclass_type (type): The type of the dataclass to load.

    Returns:
        dataclass: An instance of the dataclass loaded from the TOML file.
    """
    # 1. Read the TOML file
    with open(filepath, "r") as f:
        toml_data = tomlkit.load(f)
    
    # 2. Deserialize the TOML data into a dictionary
    data_dict = dict(toml_data)
    
    # 3. Convert to dataclass, with Combiparam casting
    dataclass_result = dacite.from_dict(
        dataclass_type,
        data_dict,
        dacite.Config(cast=[combiparam.Combiparam])
    ) 
    
    return dataclass_result

# def dict_elements_to_tuple(d: dict, ignore_iterable: bool = False) -> dict:
#     """Converts all elements in a dictionary to tuples.

#     Args:
#         d (dict): The dictionary to convert.

#     Returns:
#         dict: The dictionary with all elements converted to tuples.
#     """
#     if ignore_iterable:
#         # for the iterable values, keep them as they are
#         return {k: (v,) if not isinstance(v, (list, tuple)) else v for k, v in d.items()}
#     return {k: [v,] for k, v in d.items()}

def serialize_combiparam_dict(d: dict) -> dict:
    """Recursively deserializes a dictionary with Combiparam values.

    This function traverses a dictionary and converts any `Combiparam`
    instance to its underlying list of values. It handles nested dictionaries
    recursively.

    Args:
        d (dict): The dictionary to deserialize.

    Returns:
        dict: The deserialized dictionary.
    """
    deserialized_dict = {}
    for key, value in d.items():
        if isinstance(value, combiparam.Combiparam):
            deserialized_dict[key] = value._values
        elif isinstance(value, dict):
            deserialized_dict[key] = serialize_combiparam_dict(value)
        else:
            deserialized_dict[key] = value
    return deserialized_dict
