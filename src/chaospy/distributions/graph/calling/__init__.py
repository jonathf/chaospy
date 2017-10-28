# pylint: disable=wildcard-import
"""Run through network to perform an operator."""
from .dep import dep_call
from .fwd import fwd_call
from .inv import inv_call
from .mom import mom_call
from .pdf import pdf_call
from .range_ import range_call
from .rnd import rnd_call
from .ttr import ttr_call
from .val import val_call


CALL_FUNCTIONS = {
    "dep": dep_call,
    "fwd": fwd_call,
    "inv": inv_call,
    "mom": mom_call,
    "pdf": pdf_call,
    "range": range_call,
    "rnd": rnd_call,
    "ttr": ttr_call,
    "val": val_call,
}
