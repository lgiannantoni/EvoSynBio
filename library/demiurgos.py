import logging
import random
import time
from copy import deepcopy

import numpy as np
from cosimo.simulator import *
from cosimo.utils import AdvEnum

from library.common.geometry import Shape
from library.common.utils import InputOutput

logging.getLogger(__name__).setLevel(logging.DEBUG)

# TODO generalize the solution: find the min radius of the circumference on which K points can be found
# that are equally spaced, each k-th point having integer coordinates (cos(2kpi/K), sin(2kpi/K)).
# Note that Schinzel circles are not ideal.
# Here is a dumb solution with K=8 and no movement about z axis
DRIFT_DIRS = [(2, 0, 0), (1, 1, 0), (0, 2, 0), (-1, 1, 0), (-2, 0, 0), (-1, -1, 0), (0, -2, 0), (1, -1, 0),
              (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]  # aggiunte rispetto alle 8 direzioni già scelte sopra
GF_HIGH_COUNT = 30
GF_LOW_COUNT = 5
TRAIL_COUNT = 10
PLACING_RADIUS = 5
PLACING_HEIGHT = 1

#TODO leggere i segnali da GRID_CONFIG -> GRID_SIGNALS, come fatto in GridSimulator
class Signal(AdvEnum):
    GF = 0
    Trail = 1

class Demiurgos(ISimulator):
    _t: int
    _protocol: dict
    _placing: list
    _protocol_step: int
    _placing_coord: list

    def __init__(self):
        super().__init__([InputOutput.CONFIG], [], [], "Demiurgos")
        self.reset()

    @Pyro4.expose
    def reset(self):
        self._t = 0
        self._protocol = dict()
        self._placing_coord = list()

    @Pyro4.expose
    def step(self, *args, **kwargs) -> (tuple, dict):
        super(self.__class__, self).step(**kwargs)
        if not kwargs or not all([_in.name in kwargs.keys() for _in in self.input_list]):
            raise ValueError(f"The module requires this input: {self.input_list}.")

        if InputOutput.PROTOCOL.name in kwargs:
            self._protocol_step = int(kwargs[InputOutput.CONFIG.name][InputOutput.UGP_CONFIG.name]["PROTOCOL_STEP"])
            del kwargs[InputOutput.CONFIG.name][InputOutput.UGP_CONFIG.name]
            self._protocol = deepcopy(kwargs[InputOutput.PROTOCOL.name])
            del kwargs[InputOutput.PROTOCOL.name]

        if InputOutput.PLACING.name in kwargs:
            self._placing_coord = kwargs[InputOutput.PLACING.name]
            del kwargs[InputOutput.PLACING.name]
            _shapes = {"CYLINDER " + str(_i): {"CENTER": _coord, "RADIUS": PLACING_RADIUS, "HEIGHT": PLACING_HEIGHT} for _i, _coord in enumerate(self._placing_coord)}
            _coords = Shape.make_shapes(**_shapes)
            _coords = [_coord for _coord in _coords if all(_c >= 0 for _c in _coord)]
            _cells_coord = [("CELL", _c) for _c in _coords]
            if "GRID_ADD" not in kwargs[InputOutput.CONFIG.name][InputOutput.GRID_CONFIG.name]:
                kwargs[InputOutput.CONFIG.name][InputOutput.GRID_CONFIG.name]["GRID_ADD"] = list()
            kwargs[InputOutput.CONFIG.name][InputOutput.GRID_CONFIG.name]["GRID_ADD"].extend(_cells_coord)

        if self._t in self._protocol and self._protocol[self._t]:  # skip when protocol[t] is empty
            print(f"Demiurgos: processing protocol")
            grid_signals = list()
            for cmd in self._protocol[self._t]:
                tag, opt, coord = cmd.split(' ')
                assert tag in Signal.names(), f"Expected Signals are {Signal.names()}, {tag} received."
                assert type(eval(coord)) == tuple
                if tag == Signal.GF.name:
                    _gf_count = GF_HIGH_COUNT if opt == "HIGH" else GF_LOW_COUNT
                    _drift_dirs = [DRIFT_DIRS[_i] for _i in np.random.choice(range(len(DRIFT_DIRS)), size=_gf_count, replace=True)]
                    for k in range(_gf_count):
                        s = ("SIGNAL", eval(coord), {"TYPE": tag, "DRIFT_DIRECTION": _drift_dirs[k]})
                        grid_signals.append(s)
                elif tag == Signal.Trail.name:
                    _trail_count = TRAIL_COUNT
                    _drift_dirs = [DRIFT_DIRS[_i] for _i in np.random.choice(range(len(DRIFT_DIRS)), size=_trail_count, replace=True)]
                    for k in range(_trail_count):
                        s = ("SIGNAL", eval(coord), {"TYPE": tag, "DRIFT_DIRECTION": _drift_dirs[k]})
                        grid_signals.append(s)

            kwargs[InputOutput.DEMIURGOS_EVENTS.name] = {}
            kwargs[InputOutput.DEMIURGOS_EVENTS.name]["GRID_ADD"] = grid_signals
            #logging.debug(f"Demiurgos: @{self._t}: {grid_signals}")
        else:
            try:
                del kwargs[InputOutput.DEMIURGOS_EVENTS.name]
            except:
                pass

        self._t += 1

        return args, kwargs

    @Pyro4.expose
    def data(self):
        return {}

    def add_model(self, *args, **kwargs):
        raise NotImplementedError

    def remove_model(self, *args, **kwargs):
        # TODO
        pass

if __name__ == "__main__":
    # main_prova()
    import sys

    print(sys.argv)
    assert (len(sys.argv) > 2)
    host = sys.argv[1]
    port = int(sys.argv[2])
    print(f"port {port}")
    kwargs = {"host": host, "port": port}
    Demiurgos.serve(**kwargs)