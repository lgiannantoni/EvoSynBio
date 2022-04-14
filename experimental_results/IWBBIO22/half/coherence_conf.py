from library.common import geometry
from library.lifecycle.data.regan_exp_data import REGAN_INIT_STATES

META = {
    'Demiurgos': {
        'python': 'library.demiurgos:Demiurgos',
        # 'remote': 'user@machine.domain:4242'
    },
    'GridSimulator': {
        'python': 'library.space.simple_grid:GridSimulator',
    },
    'SignalTranslator': {
        'python': 'library.signal_translator:SignalTranslator',
    },
    'LifecycleSimulator': {
        'python': 'library.lifecycle.models.bool_lifecycle:LifecycleSimulator',
    },
    'Collector': {
        'python': 'library.data_collector.collector:Collector',
    }
}

GRID_CONFIG = {"GRID_DIM": (200, 200, 200),
               "GRID_TIMESCALE": 1,
               "GRID_SIGNALS": ['GF', 'Trail'],
               "GRID_ADD_WHERE": [
                   [{"CUBOID": {"WIDTH": 200, "DEPTH": 200, "HEIGHT": 3, "ORIGIN": [0, 0, 0]}}, "ECM"],
               ],
               "MIN_Z": 3,
               "GRID_DRAW_INTERVAL": 20
               }
LIFECYCLE_CONFIG = {"LIFECYCLE_INIT_STATE": REGAN_INIT_STATES.PROLIFERATING_1.value,
                    "LIFECYCLE_TIMESCALE": 1,
                    "LIFECYCLE_DRAW_INTERVAL": 1,
                    "LIFECYCLE_MANAGE_DEATH": True,
                    }

SIM_CONFIG = {
    "CONFIG": {
        "SIM_STEPS": 1500,
        "GRID_CONFIG": GRID_CONFIG,
        "LIFECYCLE_CONFIG": LIFECYCLE_CONFIG,
    }
}

HALF = {"DESCR": {"CUBOID": {"width": 200, "DEPTH": 100, "HEIGHT": 1, "ORIGIN": (0, 0, 3)}},
        "BOUNDING_BOX": ((0, 200), (0, 100), (3, 4))}

UGP_CONFIG = {
    "PROTOCOL_STEP": 5,
    "TARGET": HALF,
    "MASK": geometry.get_mask(GRID_CONFIG["GRID_DIM"], HALF["DESCR"])
}
