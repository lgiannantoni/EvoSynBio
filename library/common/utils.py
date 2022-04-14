import cosimo
from cosimo.utils import AdvEnum


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Objectless:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError('%s should not be instantiated' % cls)


# TODO, YESTERDAY! Check InputOutput usage in CoSimo and rename this one (overlappings...)
class InputOutput(AdvEnum):
    DEBUG = -8
    NONE = -2
    ALL = -1
    PATH = 0
    DEBUG_PATH = 42
    OUTPUT_PATH = 41
    CONFIG = 1
    META = -42
    SIM_STEPS = -666
    STOP_SIMULATION = 999
    DIGEST = 2
    PLACING = 3
    STIMULI = 4
    GRID_CONFIG = 6
    GRID_EVENTS = 7
    LIFECYCLE_CONFIG = 8
    LIFECYCLE_EVENTS = 9
    DEMIURGOS_EVENTS = -77
    SIGNAL_TRANSLATOR_EVENTS = 666
    DIFFERENTIATION_CONFIG = 555
    DIFFERENTIATION_EVENTS = 444
    UGP_CONFIG = -13
    PROTOCOL = 10


# questo potrebbe andare dentro la classe del modello del differenziamento
class DiffStates(AdvEnum):
    PLURIPOTENT_STEM_CELL = 0
    SURFACE_EPITHELIAL_COMMITMENT = 1
    KERATINOCYTE_LINEAGE_SELECTION = 2
    BASAL_KERATINOCYTE = 3
    SPINOUS_KEATINOCYTE = 4
    GRANULAR_KERATINOCYTE = 5
    APOPTOSIS = -1


class DiffStateColors(AdvEnum):
    # https://www.webucator.com/article/python-color-constants-module/
    DEFAULT = '#4169E1'  # royalblue
    PLURIPOTENT_STEM_CELL = '#9B30FF'  # purple1
    SURFACE_EPITHELIAL_COMMITMENT = '#00FF7F'  # springgreen
    KERATINOCYTE_LINEAGE_SELECTION = '#4682B4'  # steelblue
    BASAL_KERATINOCYTE = '#CD853F'  # tan3
    SPINOUS_KEATINOCYTE = '#008080'  # teal
    GRANULAR_KERATINOCYTE = '#FF6347'  # tomato1
    APOPTOSIS = '#000000' # pitch black

    @classmethod
    def getColor(cls, _diff_state):
        assert _diff_state in cls.names(), f"{_diff_state} not in {cls.__name__} names ({cls.names()})!"
        return eval(cls.__name__ + "." + _diff_state + ".value")