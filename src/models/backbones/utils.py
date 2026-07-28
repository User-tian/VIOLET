from inspect import isfunction
from typing import Callable, Optional, TypeVar, Union
from typing_extensions import TypeGuard
import collections.abc
from itertools import repeat

T = TypeVar("T")

def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None

def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
