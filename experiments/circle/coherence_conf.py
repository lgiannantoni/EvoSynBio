from library.common import geometry
from library.lifecycle.data.regan_exp_data import REGAN_INIT_STATES

META = {
    'Demiurgos': {
        'python': 'library.demiurgos:Demiurgos',
        # 'remote': 'leonardo@actarus.polito.it:4242'
    },
    'GridSimulator': {
        'python': 'library.space.simple_grid:GridSimulator',
        # 'remote': 'leonardo@actarus.polito.it:9999'
        # 'remote': 'lg@127.0.0.1:9999'
    },
    'SignalTranslator': {
        'python': 'library.signal_translator:SignalTranslator',
        # 'remote': 'leonardo@actarus.polito.it:7777'
    },
    'LifecycleSimulator': {
        'python': 'library.lifecycle.models.bool_lifecycle:LifecycleSimulator',
        # 'remote': 'leonardo@actarus.polito.it:9666'
    },
    'Collector': {
        'python': 'library.data_collector.collector:Collector',
        # 'remote': 'leonardo@actarus.polito.it:6666'
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

CIRCLE = {"DESCR": {"CYLINDER": {"center": (100, 100, 3), "radius": 60, "height": 1}},
               "BOUNDING_BOX": ((40, 160), (40, 160), (3, 4))}

UGP_CONFIG = {
    "PROTOCOL_STEP": 5,
    "TARGET": CIRCLE,
    "MASK": geometry.get_mask(GRID_CONFIG["GRID_DIM"], CIRCLE["DESCR"])
}
