"""
Combiparam module for handling parameter combinations in automation tasks.
A combiparam is a encapsulation of a Python list, but specifically marked as involving in parameter combinations. On the other hand, a normal list will not be combined with other lists.
A combiparam is also a stationary object and should not be modified after creation & during the experiment. Methods like `append`, `extend`, `insert`, `remove`, `pop`, and `clear` are thus not provided.
"""

import logging


class Combiparam:
    
    def __init__(self, values: list):
        """
        Initializes a Combiparam instance.

        Args:
            values (list): The list of values for the parameter combination.
        """
        if not isinstance(values, list):
            raise TypeError("Values must be a list.")
        self._values = values
    
    def __repr__(self):
        """
        Returns a string representation of the Combiparam instance.
        """
        return f"Combiparam({self._values})"
    
    def __iter__(self):
        """
        Returns an iterator over the values of the Combiparam instance.
        """
        return iter(self._values)
    
    def __len__(self):
        """
        Returns the length of the values list.
        """
        return len(self._values)
    
    def __getitem__(self, index):
        """
        Returns the item at the specified index from the values list.
        
        Args:
            index (int): The index of the item to retrieve.
        
        Returns:
            The item at the specified index.
        """
        return self._values[index]

    def __eq__(self, other):
        """
        Checks if two Combiparam instances are equal.
        
        Args:
            other (Combiparam): The other Combiparam instance to compare with.
        
        Returns:
            bool: True if both instances have the same values, False otherwise.
        """
        if not isinstance(other, Combiparam):
            return False
        return self._values == other._values