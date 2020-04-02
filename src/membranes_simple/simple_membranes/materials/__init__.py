# -*- coding: utf-8 -*-
import dolfin

_MATERIALS = {}

def add_material(name, material_class):
    """
    Register a material model
    """
    _MATERIALS[name] = material_class


def register_material(name):
    """
    A class decorator to register material models
    """

    def register(material_class):
        add_material(name, material_class)
        return material_class

    return register


def get_material(name):
    """
    Return a material model by name
    """
    try:
        return _MATERIALS[name]
    except KeyError:
        raise


class Material(object):
    pass

from . import materials