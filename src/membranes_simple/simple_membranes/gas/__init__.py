# -*- coding: utf-8 -*-
import dolfin

_GAS_LAWS = {}

def add_gas_law(name, gas_law_class):
    """
    Register a gas_law model
    """
    _GAS_LAWS[name] = gas_law_class


def register_gas_law(name):
    """
    A class decorator to register gas_law models
    """

    def register(gas_law_class):
        add_gas_law(name, gas_law_class)
        return gas_law_class

    return register


def get_gas_law(name):
    """
    Return a gas_law model by name
    """
    try:
        return _GAS_LAWS[name]
    except KeyError:
        raise

from . import gas_laws