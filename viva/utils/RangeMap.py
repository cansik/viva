from dataclasses import dataclass
from typing import Optional, TypeVar, Generic

from sortedcontainers import SortedDict

T = TypeVar('T')


@dataclass
class RangeResult(Generic[T]):
    start: int
    end: int
    value: T


class RangeMap(Generic[T]):
    """
    A data structure that maps ranges [start, end) to values. Supports efficient range lookups.

    Attributes:
        ranges (SortedDict): A sorted dictionary where keys are the start of ranges and values are tuples of (end, value).
    """

    def __init__(self) -> None:
        """Initialize an empty RangeMap."""
        self.ranges: SortedDict[int, tuple[int, T]] = SortedDict()

    def add_range(self, start: int, end: int, value: T) -> None:
        """
        Add a range [start, end) mapped to a specific value.

        Args:
            start (int): The start of the range (inclusive).
            end (int): The end of the range (exclusive).
            value (T): The value associated with the range.

        Raises:
            ValueError: If start >= end.
        """
        if start >= end:
            raise ValueError("Start of range must be less than end")
        self.ranges[start] = (end, value)

    def get(self, key: int) -> Optional[RangeResult[T]]:
        """
        Retrieve the range result containing the key.

        Args:
            key (int): The key to look up.

        Returns:
            Optional[RangeResult[T]]: A dataclass containing start, end, and value of the range, or None if no range contains the key.
        """
        index = self.ranges.bisect_right(key) - 1
        if index < 0:
            return None
        start = self.ranges.keys()[index]
        end, value = self.ranges[start]
        if start <= key < end:
            return RangeResult(start=start, end=end, value=value)
        return None

    def clear(self) -> None:
        """Remove all ranges from the RangeMap."""
        self.ranges.clear()

    def __getitem__(self, key: int) -> Optional[RangeResult[T]]:
        """
        Retrieve the range result containing the key using the [] operator.

        Args:
            key (int): The key to look up.

        Returns:
            Optional[RangeResult[T]]: A dataclass containing start, end, and value of the range, or None if no range contains the key.
        """
        return self.get(key)
