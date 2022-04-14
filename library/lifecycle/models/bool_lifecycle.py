# boolean.py
"""

"""
import logging
import operator
import os
import random
from typing import List, Dict, Any

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from anytree import RenderTree
from cosimo.simulator import *
from cosimo.utils import AdvEnum

from library.common.utils import InputOutput
#from library.lifecycle.models.bool_models import DifferentiationModel as Model
from library.lifecycle.models.bool_models import Regan2020Model as Model


# TODO add anoikis, g0, g1, ... to recognized states (Model.OUTPUTS)?

class LifecycleOpt(AdvEnum):
    # LIFECYCLE_BNET_FILE = 0
    LIFECYCLE_INIT_STATE = 1
    LIFECYCLE_TIMESCALE = 2
    LIFECYCLE_DRAW_INTERVAL = 3
    LIFECYCLE_MANAGE_DEATH = 4


class LifecycleSimulator(ISimulator):
    """Simulates a number of ``Model`` models and collects some data."""

    _n_cells: int
    feed: Any  # % of cells to be fed with protocol at each step
    init_state: Any
    _t: int
    _timescale: int
    _draw_interval: int
    models: Dict[int, Model]
    dead_models: Dict[int, Model]
    _events: Any  # TODO
    plot: bool
    _manage_death: bool

    class Event(AdvEnum):
        # keys for reading simple_grid input
        SPAWNED = 2
        # REMOVED = 3
        ECM_LOSS = 4
        CELLDENSITY_HIGH = 5
        CELLDENSITY_LOW = 6
        # keys for writing output events
        REPLICATION = 0
        DEATH = 1
        # other signals from grid (as of April 22: GF, Trail)
        SIGNALS = -1

    def __init__(self):
        super().__init__(_in=[InputOutput.CONFIG, InputOutput.GRID_EVENTS], _remove=[], _add=[InputOutput.LIFECYCLE_EVENTS], name="LifecycleSimulator")
        # bnet_file, n_cells, init_state=None, duration=None, ext_driver=False,
        #                  protocol: Dict[AnyStr, int] = None, feed: Dict[int, int] = None, plot=False
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        self.reset()

    @Pyro4.expose
    def reset(self):
        logging.info(f"Resetting {self.__class__.__name__}")
        self._n_cells = 0
        # self.duration = 0
        # self._ext_driver = None
        # self.protocol = None
        self.feed = None  # % of cells to be fed with protocol at each step
        self.init_state = None
        self._t = 0
        self._timescale = 1
        self._draw_interval: int
        self.models: Dict[int, Model]
        self.dead_models: Dict[int, Model] = dict()
        self._events = None
        self.plot = False
        random.seed(a="This is a seed")

    def _setup(self, **kwargs):
        logging.info(f"{self.__class__.__name__}: setup...")
        if not all([k in kwargs.keys() for k in LifecycleOpt.names()]):
            raise ValueError(
                f"Module setup requires parameters {LifecycleOpt.names()} in {InputOutput.LIFECYCLE_CONFIG} input. Got {kwargs.keys()}.")
        # Model.reset()  # TODO check; probabilmente non necessario
        self.init_state = kwargs[LifecycleOpt.LIFECYCLE_INIT_STATE.name]
        self._timescale = kwargs[LifecycleOpt.LIFECYCLE_TIMESCALE.name]
        self._draw_interval = kwargs[LifecycleOpt.LIFECYCLE_DRAW_INTERVAL.name]
        self._manage_death = kwargs[LifecycleOpt.LIFECYCLE_MANAGE_DEATH.name]
        self.models = dict()
        # self.protocol = None

    @property
    def all_events(self) -> Dict[Dict[List, None], None]:
        # returns {0: {'event A': [1, 2, 3], 'event B': []}, 1: {'event A': [0], 'event B': [1]}, ...}
        return self._events

    @property
    def events(self) -> Dict[List, None]:
        # returns {0: {'event A': [1, 2, 3], 'event B': []}
        if self._t > 0:
            return self._events[self._t - 1]
        return dict()

    def run(self):
        pass
        # logging.info(f"{self.__class__.__name__} -- Creating entities.")
        # for i in range(self._n_cells):
        #     self.add_model()
        # logging.info(f"{self.__class__.__name__} -- {self._n_cells} entities created.")
        #
        # logging.info(f"{self.__class__.__name__} -- Simulation start.")
        # self._events = dict()
        # for _step in range(self.duration):
        #     _, kwargs = self.step()
        #     stop_simulation = kwargs[InputOutput.BOOL_SIM_STOP.name]
        #     self._events[_step] = kwargs[InputOutput.BOOL_SIM_EVENTS.name]
        #     if stop_simulation:
        #         break
        # logging.info(f"{self.__class__.__name__} -- Simulation end.")

    def add_model(self, _id: int = None, parent_id: int = None):
        """Create an instance of ``Model`` with *init_val*."""
        parent = self.models[parent_id] if parent_id else None
        model = Model(self._t, _id=_id, init_state=self.init_state, parent=parent)
        self.models[model.id] = model

    def remove_model(self, _id: int):
        self.dead_models[_id] = self.models[_id]
        del self.models[_id]
        assert _id not in self.models.keys(), f"{self.__class__.__name__} -- Model {_id} should not be in self.models"

    @Pyro4.expose
    def step(self, draw=False, *args, **kwargs) -> (tuple, dict):  # (bool, Dict[List, None]):
        """Perform a simulation step."""
        super(self.__class__, self).step(**kwargs)
        if kwargs:
            if not all([_in.name in kwargs.keys() for _in in self.input_list]):
                raise ValueError(f"The module requires this input: {self.input_list}.")

        if self._t == 0:
            self._setup(**kwargs[InputOutput.CONFIG.name][InputOutput.LIFECYCLE_CONFIG.name])
            del kwargs[InputOutput.CONFIG.name][InputOutput.LIFECYCLE_CONFIG.name]  # config done, remove info

        # 0. remove dead cells if death is managed by someone else (as of now: Differentiation FSM)
        if InputOutput.LIFECYCLE_EVENTS.name in kwargs:
            if not self._manage_death:
                lifecycle_events = kwargs[InputOutput.LIFECYCLE_EVENTS.name]
                if Model.Actions.DEATH.name in lifecycle_events:
                    print(f"@{self._t} {self.name}: removing {len(lifecycle_events[Model.Actions.DEATH.name])} dead cells")
                    for _id in lifecycle_events[Model.Actions.DEATH.name]:
                        self.remove_model(_id)

            del kwargs[InputOutput.LIFECYCLE_EVENTS.name]

        events = {k.name: list() for k in list(Model.Actions)}
        # this is different because each cell can be in a different set of "overlapping" states
        # events[Model.Actions.DIFFERENTIATION.name] = dict()

        # TODO if GRID_EVENTS in kwargs and SPAWNED in...
        # 1. add new cells spawned by grid with assigned id
        for new_cell_id, parent_id in kwargs[InputOutput.GRID_EVENTS.name][LifecycleSimulator.Event.SPAWNED.name].items():
            self.add_model(new_cell_id, parent_id)
        # 2. set ECM and Stiff_ECM to 0 if grid signals ECM loss
        if kwargs[InputOutput.GRID_EVENTS.name][LifecycleSimulator.Event.ECM_LOSS.name]:
            for cell_id in kwargs[InputOutput.GRID_EVENTS.name][LifecycleSimulator.Event.ECM_LOSS.name]:
                self.models[cell_id].ecm_loss = True
        # 3. set new cell_density according to grid data
        if kwargs[InputOutput.GRID_EVENTS.name][LifecycleSimulator.Event.CELLDENSITY_LOW.name]:
            # print(f"setting cell density low to cells {kwargs[InputOutput.GRID_EVENTS.name][LifecycleSimulator.Event.CELLDENSITY_LOW.name]}")
            for cell_id in kwargs[InputOutput.GRID_EVENTS.name][LifecycleSimulator.Event.CELLDENSITY_LOW.name]:
                if cell_id in self.models.keys():
                    self.models[cell_id].cell_density_low = True
                    self.models[cell_id].cell_density_high = False
        if kwargs[InputOutput.GRID_EVENTS.name][LifecycleSimulator.Event.CELLDENSITY_HIGH.name]:
            #print(f"setting cell density low to cells {kwargs[InputOutput.GRID_EVENTS.name][LifecycleSimulator.Event.CELLDENSITY_HIGH.name]}")
            for cell_id in kwargs[InputOutput.GRID_EVENTS.name][LifecycleSimulator.Event.CELLDENSITY_HIGH.name]:
                if cell_id in self.models.keys():
                    self.models[cell_id].cell_density_high = True
                    self.models[cell_id].cell_density_low = False

        protocol = dict()
        if kwargs[InputOutput.SIGNAL_TRANSLATOR_EVENTS.name]:
            protocol = kwargs[InputOutput.SIGNAL_TRANSLATOR_EVENTS.name]
            del kwargs[InputOutput.SIGNAL_TRANSLATOR_EVENTS.name]
        else:
            # TODO check
            protocol = {_id: {'GF': 0, 'GF_High': 0, 'Trail': 0} for _id in self.models.keys()}
        # 4. check lifecycle macro-states for each model
        for id_, model in self.models.copy().items():
            # 2.1 model should replicate itself: tell spatial grid
            # if Model.is_true(cond={'Apoptosis': 0}, state=model.state):
            if model.Apoptosis:
                if self._manage_death:
                    if random.random() <= 0.20:
                        # if self.plot:
                        #    self.tapestry(model)
                        events[Model.Actions.DEATH.name].append(model.id)
                        self.remove_model(model.id)
                        logging.info(f"{self.__class__.__name__} -- @ {self._t} -- {model.id} died")
                        continue
                # else:
                #    # if random.random() <= 0.20:
                #    # vediamo se l'apopotosi è sufficientemente rallentata dalla macchina a stati a valle
                #    events[Model.Actions.DEATH].append(model.id)  # questo potrebbe non servire...

            # do step!
            _cmd = protocol[model.id] if model.id in protocol.keys() else {}
            assert all([_k in model.state.keys() for _k in _cmd.keys()]), f"DIOCANE!\n{set(model.state.items()) - set(_cmd.items())}\n{set(_cmd.items()) - set(model.state.items())}"
            model.step(_cmd)
            # if Model.is_true(cond={'Replication': 1}, state=model.state[self._t]):
            # if Model.is_true(cond={'Replication': 1}, state=model.state):
            if model.Replication:
                # TODO check with Roberta
                # replicate with probability p
                if True:  # random.random() <= 0.50:
                    # spawning children for next round, waiting for grid to assign uuid
                    events[Model.Actions.REPLICATION.name].append(model.id)
            if hasattr(model, 'DifferentiationState') or isinstance(getattr(type(model), 'DifferentiationState', None), property):
                ch = model.DifferentiationStateChanged
                if ch:
                    if not events[Model.Actions.DIFFERENTIATION.name]:
                        events[Model.Actions.DIFFERENTIATION.name] = dict()
                    # print(f"{model.id}: {ch}")
                    events[Model.Actions.DIFFERENTIATION.name][model.id] = ch

        # kwargs[InputOutput.BOOL_SIM_STOP.name] = True if len(self.models) == 0 else False
        kwargs[InputOutput.LIFECYCLE_EVENTS.name] = events
        logging.info(f"{self.__class__.__name__}: {len(self.models)} cells at step {self._t}")
        self._t += 1
        return args, kwargs

    @Pyro4.expose
    def data(self):
        # return {id_: m.data for id_, m in self.models.items()}
        d = dict()
        for _step in range(self._t):
            d[_step] = dict()
            for _id, _m in self.models.items():
                if _step in _m.state:
                    d[_step][_id] = dict((_k, _m.state[_step][_k]) for _k in Model.OUTPUTS)
                else:
                    d[_step][_id] = dict((_k, -1) for _k in Model.OUTPUTS)
        return d

    def tree(self):
        buf = list()
        d = {**self.models, **self.dead_models}
        for _id, _m in sorted(d.items(), key=operator.itemgetter(0))[:len(self.models)]:
            for pre, fill, node in RenderTree(_m):
                treestr = u"%s%s" % (pre, node.id)
                last_step = list(_m.state.keys())[-1]
                # TODO check
                buf.append(treestr.ljust(8) + (' alive' if 'Apoptosis' in _m.state[last_step] and _m.state[last_step][
                    'Apoptosis'] == 0 else ' dead'))
        return "\n".join(buf)

    def write_results(self, to="results.dat"):
        with open(to, 'w') as exp_results:
            logging.info(f"{self.__class__.__name__} -- Writing results to file {exp_results.name}")
            exp_results.write(f"N_CELLS {N_CELLS}\n")
            exp_results.write(f"INIT_STATE {INIT_STATE}\n")
            exp_results.write(f"PROTOCOL {PROTOCOL}\n")
            exp_results.write(f"DURATION {DURATION} steps\n")
            # exp_results.write(f"Simulation finished (duration: {i - 1} steps)\n")
            exp_results.write(f"{len(sim.models)} alive cells\n")
            exp_results.write(f"{len(sim.dead_models)} dead cells\n")
            # dd = sim.data()
            # exp_results.write(f"output {dd.items()}")
            # exp_results.write(sim.tree())
            df = pd.DataFrame(self._events).transpose()
            exp_results.write(f"Execution stopped at step {len(df) - 1} (0 cells alive)\n")
            df.to_string(exp_results)
        logging.info(f"{self.__class__.__name__} -- Done.")


def serve(argv):
    assert (len(argv) > 2), f"Expected arguments: <program> <host> <port>. Got {' '.join(argv)}"
    host = argv[1]
    port = int(argv[2])
    print(f"port {port}")
    kwargs = {"host": host, "port": port}
    LifecycleSimulator.serve(**kwargs)

if __name__ == "__main__":
    import sys
    serve(sys.argv)


# if __name__ == '__main__':
#     from library.lifecycle.data.regan_exp_data import REGAN_EXP, EXP
#
#     # This is how the simulator could be used:
#     N_CELLS = 5
#     INIT_STATE = REGAN_EXP['Regan_1_quiescent2anoikis'][EXP.init_state]
#     PROTOCOL = REGAN_EXP['Regan_1_quiescent2anoikis'][EXP.protocol]
#     # print(f"protocol {PROTOCOL}")
#     DURATION = max(PROTOCOL.keys()) + 1
#
#     # Logger.getLogger('matplotlib').setLevel(Level.ERROR)
#     # sim = Simulator(bnet_file="../data/regan2020.bnet", n_cells=N_CELLS, init_state=INIT_STATE, duration=DURATION,
#     #                     protocol=PROTOCOL, plot=False)
#
#     kwargs = {InputOutput.CONFIG.name: {
#         InputOutput.LIFECYCLE_CONFIG.name: {#"LIFECYCLE_BNET_FILE": "../data/regan2020.bnet",
#                                             "LIFECYCLE_INIT_STATE": REGAN_EXP['Regan_2_proliferation2anoikis'][EXP.init_state],
#                                             "LIFECYCLE_TIMESCALE": 1,
#                                             "LIFECYCLE_DRAW_INTERVAL": 1,
#
#         },
#     },
#     InputOutput.GRID_EVENTS.name: {'SPAWNED':
#                                        {226342273803344125728680122314229309424: None,
#                                         251400722381318013486144802793001007937: None,
#                                         206814744286531949838648585207961370790: None,
#                                         327463866859093737560666092480710577366: None,
#                                         300058428272089005060538137567039257209: None,
#                                         104105104108353130623393740016594252228: None,
#                                         203791298355420147201781396204199291413: None,
#                                         35466258178066977688270175797123321629: None,
#                                         80110063412880363238205193708478292736: None,
#                                         61450793094859548438329929150037727390: None
#                                         }
#         },
#     }
#     print(os.getcwd())
#     sim = LifecycleSimulator()
#     for _ in range(100):
#         _, kwargs = sim.step(**kwargs)
#         #print(kwargs)
