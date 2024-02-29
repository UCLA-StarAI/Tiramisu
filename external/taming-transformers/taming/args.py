import inspect
import os
from argparse import _ArgumentGroup, ArgumentParser, Namespace
from ast import literal_eval
from contextlib import suppress
from functools import wraps
from typing import Any, Callable, cast, Dict, List, Tuple, Type, TypeVar, Union

import pytorch_lightning as pl

_T = TypeVar("_T", bound=Callable[..., Any])
_ARGPARSE_CLS = Union[Type["pl.LightningDataModule"], Type["pl.Trainer"]]


def from_argparse_args(
    cls: _ARGPARSE_CLS,
    args: Union[Namespace, ArgumentParser],
    **kwargs: Any,
) -> Union["pl.LightningDataModule", "pl.Trainer"]:
    """Create an instance from CLI arguments. Eventually use variables from OS environment which are defined as
    ``"PL_<CLASS-NAME>_<CLASS_ARUMENT_NAME>"``.

    Args:
        cls: Lightning class
        args: The parser or namespace to take arguments from. Only known arguments will be
            parsed and passed to the :class:`Trainer`.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
            These must be valid Trainer arguments.

    Examples:

        >>> from lightning.pytorch import Trainer
        >>> parser = ArgumentParser(add_help=False)
        >>> parser = Trainer.add_argparse_args(parser)
        >>> parser.add_argument('--my_custom_arg', default='something')  # doctest: +SKIP
        >>> args = Trainer.parse_argparser(parser.parse_args(""))
        >>> trainer = Trainer.from_argparse_args(args, logger=False)
    """
    if isinstance(args, ArgumentParser):
        args = cls.parse_argparser(args)

    params = vars(args)

    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(cls.__init__).parameters
    trainer_kwargs = {name: params[name] for name in valid_kwargs if name in params}
    trainer_kwargs.update(**kwargs)

    return cls(**trainer_kwargs)