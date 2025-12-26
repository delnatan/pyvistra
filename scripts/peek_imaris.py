# %%
import h5py
import numpy as np
from pyvistra.imaris_reader import ImarisReader


# %%
def h5_tree(val, pre=""):
    """
    Recursive function to print the HDF5 file structure in a tree diagram.

    Usage:
    with h5py.File('my_data.h5', 'r') as f:
        h5_tree(f)

    Args:
        val: The h5py object (File or Group) to iterate over.
        pre: The prefix string for indentation (used internally).
    """
    # Get the list of items in the current group
    items = len(val)

    for key, val in val.items():
        items -= 1

        # Determine if we are at the last item to choose the correct connector
        if items == 0:
            connector = "└── "
            # If last item, child prefix adds spaces
            child_pre = pre + "    "
        else:
            connector = "├── "
            # If not last, child prefix adds a vertical bar
            child_pre = pre + "│   "

        # Distinguish between Groups and Datasets for formatting
        if isinstance(val, h5py.Group):
            print(f"{pre}{connector}📂 {key}")
            # Recursively call for the subgroup
            h5_tree(val, child_pre)

        elif isinstance(val, h5py.Dataset):
            # For datasets, print shape and dtype info
            print(
                f"{pre}{connector}📄 {key}  [Shape: {val.shape}, Type: {val.dtype}]"
            )


# %%
datfn = "/users/delnatan/Downloads/UD877_wt_03.ims"

import json

with h5py.File(datfn, "r") as h5:
    # h5_tree(h5["DataSetInfo"])
    custom_data_dict = dict(h5["DataSetInfo/CustomData"].attrs.items())
    data = {
        k: "".join(np.char.decode(vals))
        for k, vals in custom_data_dict.items()
    }
    with open("custom_data.json", "w") as fhd:
        json.dump(data, fhd, indent=2)
# %%
