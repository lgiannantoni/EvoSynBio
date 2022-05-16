import logging
import os
import random
from collections import defaultdict
from copy import deepcopy
from typing import Dict, AnyStr, List, DefaultDict

import PyBoolNet.FileExchange as FE
import PyBoolNet.InteractionGraphs as IG
import PyBoolNet.PrimeImplicants as PIs
import PyBoolNet.StateTransitionGraphs as STG
from anytree import NodeMixin
from cosimo.model import IModel
from cosimo.utils import AdvEnum
from path import Path


class BoolModel(IModel, NodeMixin):
    NAME = 'BoolModel abstract class'
    BNET: str = None
    PRIMES = None
    INPUTS: List[str] = None
    OUTPUTS: List[str] = None
    INIT_STATES: DefaultDict = None
    INIT_SUBSPACE: DefaultDict = None

    logging.getLogger(__name__).setLevel(logging.DEBUG)

    def __init__(self, step: int, _id: int, init_state: Dict[str, int] = None, parent: int = None,
                 children: List[int] = None):
        super().__init__()
        assert not (self.__class__.BNET is None and self.__class__.PRIMES is None), "Both Model.BNET and Model.PRIMES are None"
        if self.__class__.PRIMES is None:
            self.__class__._make_primes()
        self.id: int = _id
        self.parent = parent
        self.children = children if children is not None else list()
        self.state: Dict[str, int] = dict()
        if self.parent:
            self.state = deepcopy(self.parent.state)
        elif init_state:
            self.state = init_state
        else:
            self.state = self._init_state()
        self._step = step

    @classmethod
    def reset(cls):
        cls.PRIMES = None

    def step(self, _input={}):
        self._step += 1
        self.state = STG.successor_synchronous(self.__class__.PRIMES, dict(list(self.state.items()) + list(_input.items())))

    @classmethod
    def _make_primes(cls):
        logging.info(f"{cls.__name__} -- Building primes")
        assert not (cls.BNET is None and cls.PRIMES is None), "Both bnet_file and Model.PRIMES are None"
        cls.PRIMES = FE.bnet2primes(cls.BNET)

    @classmethod
    def _generate_init_states(cls, cond: Dict[AnyStr, int]):
        max_attempts = 3  # 2 ** (len(cls.INPUTS) + 1)
        logging.info(f"{cls.__name__} -- Generating initial states (max_attempts {max_attempts})")
        for i in range(max_attempts):
            s = STG.random_state(cls.PRIMES, subspace=cond)
            cls.INIT_STATES[hash(frozenset(s.items()))] = s

    @classmethod
    def _init_state(cls):
        if cls.INIT_STATES:
            return random.choice(list(cls.INIT_STATES.values()))
        elif cls.INIT_SUBSPACE:
            cls._generate_init_states(cond=cls.INIT_SUBSPACE)  # {'Casp3': 0, 'CAD': 0, 'Apoptosis': 0})
            return random.choice(list(cls.INIT_STATES.values()))
        else:
            return STG.random_state(cls.PRIMES)

    # TODO check state and remove id
    @staticmethod
    def is_true(cond: Dict[AnyStr, int], state: Dict[AnyStr, int]):
        for _k in cond.keys():
            if _k not in state.keys():
                logging.error(f"{BoolModel.__name__} -- Key {_k} not found in state keys.")
                pass
        assert all([_s in state.keys() for _s in cond.keys()]), f"Not all {cond.keys()} in {state.keys()}"
        return all([state[_k] == _v for _k, _v in cond.items()])

    @staticmethod
    def is_false(cond: Dict[AnyStr, int], state: Dict[AnyStr, int]):
        return not BoolModel.is_true(cond, state)

    @classmethod
    def draw(cls, name=None, path="."):
        fname = str(name) + "_" + cls.NAME if name is not None else cls.NAME
        IG.create_image(cls.PRIMES, os.path.join(path, fname + ".pdf"))


class Regan2020Model(BoolModel):
    """
    Boolean model based on Regan 2020 paper
    """

    class Actions(AdvEnum):
        DEATH = 0
        REPLICATION = 1

    NAME = 'Regan2020'
    BNET = Path(os.path.dirname(__file__)) / "../data/regan2020.bnet"
    PRIMES = None
    INPUTS = ['GF', 'GF_High', 'Trail', 'ECM', 'Stiff_ECM', 'CellDensity_Low', 'CellDensity_High']
    OUTPUTS = ["Apoptosis", "Replication"]
    INIT_STATES = defaultdict()
    INIT_SUBSPACE = {'GF_High': 0, 'GF': 1, 'Trail': 0, 'ECM': 1, 'Stiff_ECM': 1, 'CellDensity_Low': 1,
                     'CellDensity_High': 0, 'CAD': 0, 'Casp3': 0}  # in Regan2020

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _make_primes(cls):
        super()._make_primes()
        PIs.create_variables(cls.PRIMES, {'Apoptosis': '!ECM | (Casp3 & CAD)'})

    @property
    def Apoptosis(self):
        return self.state['Apoptosis'] == 1

    @property
    def Replication(self):
        return self.state['Replication'] == 1

    def step(self, _input={}):
        """Perform a simulation step."""
        super().step(_input)

    # TODO check...
    @property
    def ecm_loss(self):
        return self.state['ECM'] == 0 and self.state['Stiff_ECM'] == 0

    @ecm_loss.setter
    def ecm_loss(self, val: bool = True):
        self.state['ECM'] = 0 if val else 1
        self.state['Stiff_ECM'] = 0 if val else 1

    @property
    def cell_density_low(self):
        return not self.cell_density_high and self.state['CellDensity_Low'] == 1

    @cell_density_low.setter
    def cell_density_low(self, val: bool = True):
        self.cell_density_high = not val

    #TODO check...
    @property
    def cell_density_high(self):
        return self.state['CellDensity_Low'] == 1 and self.state["CellDensity_High"] == 1

    @cell_density_high.setter
    def cell_density_high(self, val: bool = True):
        self.state['CellDensity_Low'] = 1
        self.state['CellDensity_High'] = 1 if val else 0


def regan_model_testing():
    m = Regan2020Model(5, _id=226342273803344125728680122314229309424, init_state=None, parent=None)
    m.draw()
    for _ in range(3):
        m.step()
