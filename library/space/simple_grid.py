import itertools
import logging
import operator
import os
import random
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from time import sleep
from typing import Tuple, List, NamedTuple, Union, Dict, Optional, Any
from collections import defaultdict

import Pyro4
import numpy as np
from cosimo.model import IModel
from cosimo.simulator import ISimulator
from cosimo.utils import AdvEnum, UniqueIdMap

from library.common.geometry import Shape

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from library.common.utils import InputOutput, DiffStateColors, DiffStates

Tree = lambda: defaultdict(Tree)

logging.getLogger(__name__).setLevel(logging.DEBUG)


class SimpleGridOpt(AdvEnum):
    GRID_TIMESCALE = 0
    GRID_DIM = 2
    GRID_SIGNALS = 3
    GRID_ADD = 4
    GRID_ADD_WHERE = 5
    GRID_DRAW_INTERVAL = 6


class Coord(NamedTuple):
    x: int
    y: int
    z: int

    @property
    def tuple(self):
        return (self.x, self.y, self.z)

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"


class GridObjectType(AdvEnum):
    EMPTY = 0
    ECM = 1
    CELL = 2
    SIGNAL = 3


class GridObject(ABC):
    unique_id_map = UniqueIdMap()

    @classmethod
    def unique_id(cls, obj):
        """Produce a unique integer id for the object.

        Object must be *hashable*. Id is a UUID and should be unique
        across Python invocations.

        """
        return cls.unique_id_map[obj].int

    @abstractmethod
    def __init__(self, ftype: str, coord: Tuple[int, int, int] = None):
        self._uuid: int = GridObject.unique_id(self) if ftype != GridObjectType.SIGNAL.name else None
        self._coord = Coord(*coord) if coord is not None else None
        self._type: str = ftype
        self._color = "black"

    @staticmethod
    def new(ftype: str, coord: Tuple[int, int, int], **_kwargs):
        for t in list(GridObjectType.names()):
            if t == ftype:
                constr = t[:1] + t[1:].lower()
                return eval(constr)(coord, **_kwargs)
        raise TypeError("Grid can accept only {} object".format([t[:1] + t[1:].lower() for t in GridObjectType.names()]))

    def __str__(self):
        return f"{self.__class__.__name__} (uuid {self._uuid}) at {self._coord}"

    def __repr__(self):
        return f"{self.__class__.__name__} (uuid {self._uuid}) at {self._coord}"

    @property
    def type(self) -> str:
        return self._type

    @property
    def uuid(self) -> int:
        return self._uuid

    @property
    def coord(self) -> Coord:
        return self._coord

    @coord.setter
    def coord(self, coord: Tuple[int, int, int]):
        self._coord = coord

    @property
    def color(self) -> str:
        return self._color


class Cell(GridObject):
    class Opt(AdvEnum):
        RADIUS = 0

    def __init__(self, coord: Tuple[int, int, int] = None, diff_state=DiffStates.PLURIPOTENT_STEM_CELL.name, **_kwargs):
        super().__init__(ftype=GridObjectType.CELL.name, coord=coord)
        if Cell.Opt.RADIUS.name in _kwargs:
            self._radius: int = _kwargs[Cell.Opt.RADIUS.name]
        else:
            self._radius = 1
        # self._color = DiffStateColors.DEFAULT.value
        self._diff_state = diff_state
        self._color = DiffStateColors.getColor(diff_state)

    @property
    def radius(self) -> int:
        return self._radius

    @radius.setter
    def radius(self, val: int):
        assert val > 0, "Radius value must be a positive integer"
        self._radius = val

    @property
    def diff_state(self) -> str:
        return self._diff_state

    @diff_state.setter
    def diff_state(self, val: str):
        assert val in DiffStates.names(), f"{val} not in DiffStates ({DiffStates.names()})!"
        self._diff_state = val
        self._color = DiffStateColors.getColor(val)


class Ecm(GridObject):
    class Opt(AdvEnum):
        STIFF = 0

    def __init__(self, coord: Tuple[int, int, int], **_kwargs):
        super().__init__(ftype=GridObjectType.ECM.name, coord=coord)
        if Ecm.Opt.STIFF.name in _kwargs:
            self._stiff: bool = _kwargs[Ecm.Opt.STIFF.name]
        else:
            self._stiff = True
        self._color = "gainsboro"

    @property
    def stiff(self) -> bool:
        return self._stiff

    @stiff.setter
    def stiff(self, val: bool):
        self._stiff = val


class _SignalType(AdvEnum):
    pass


class Signal(GridObject):
    class Opt(AdvEnum):
        TYPE = 0
        DRIFT_DIRECTION = 1
        # DRIFT_DELAY = 2
        # DRIFT_COUNTER = 3

    SignalType: _SignalType = None
    _signal_type: SignalType
    _drift_direction: Tuple[int, int, int] = None
    _drift_delay: int = None
    _drift_counter: int = None
    _coord: Tuple[int, int, int]
    _keepalive: int

    def __init__(self, coord: Tuple[int, int, int], **kwargs):
        super().__init__(ftype=GridObjectType.SIGNAL.name, coord=coord)
        if Signal.Opt.TYPE.name in kwargs:
            if kwargs[Signal.Opt.TYPE.name] in list(Signal.SignalType.names()):
                self._signal_type = kwargs[Signal.Opt.TYPE.name]
            else:
                raise TypeError("Signal.Opt.TYPE value can accept only Signal.SignalType objects")
        else:
            self._signal_type = Signal.SignalType.random().name
        self._coord = coord
        # voglio che tuttie e 3 le opzioni sul drift siano presenti contemporaneamente in kwargs, altrimenti come se nessuna
        # if all(item in kwargs for item in Signal.Opt.names()):
        #     assert kwargs[Signal.Opt.DRIFT_DELAY.name] >= 0 and kwargs[Signal.Opt.DRIFT_COUNTER.name] >= 0,\
        #         f"{Signal.Opt.DRIFT_DELAY.name} and {Signal.Opt.DRIFT_COUNTER.name} must be both greater than or equal to zero."
        #     self._drift_delay = kwargs[Signal.Opt.DRIFT_DELAY.name]
        #     self._drift_counter = kwargs[Signal.Opt.DRIFT_COUNTER.name]
        #     self._drift_direction = kwargs[Signal.Opt.DRIFT_DIRECTION.name]

        # versione semplificata con il solo parametro _DIRECTION richiesto, e DRIFT_DELAY e _COUNTER settati direttamente qui
        if Signal.Opt.DRIFT_DIRECTION.name in kwargs:
            self._drift_direction = kwargs[Signal.Opt.DRIFT_DIRECTION.name]
            self._drift_delay = 1  # do drift every 1 simulation step, at first
            self._drift_counter = 0
            self._keepalive = 10  # TODO parametrizzare

    def can_drift(self) -> bool:
        return self._drift_direction and (self._drift_counter >= self._drift_delay)

    def step(self, new_coord, new_drift_dir = None):
        self._keepalive -= 1
        if new_coord:
            self._drift_counter = 0
            self._drift_delay += 1  # slow down drifting events every time they do happen
            self._coord = new_coord
            self._drift_direction = new_drift_dir
        else:
            self._drift_counter += 1

    @classmethod
    def set_types(cls, signals: List[str]):
        cls.SignalType = _SignalType('SignalType', signals)

    @property
    def signal_type(self) -> str:
        return self._signal_type

    @property
    def keepalive(self) -> bool:
        return self._keepalive > 0

    def __str__(self):
        return f"{self.__class__.__name__} {self._signal_type} at {self._coord}"

    def __repr__(self):
        return f"{self.__class__.__name__} {self._signal_type} at {self._coord}"


class Singleton(type(IModel)):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Grid(IModel, metaclass=Singleton):
    _scaffold_grid: Dict[Tuple[int, int, int], Union[Cell, Ecm]]
    _medium_grid: Dict[Tuple[int, int, int], List[Signal]]
    _reverse_cell_grid: Dict[int, Tuple[int, int, int]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if SimpleGridOpt.GRID_DIM.name not in kwargs:
            raise TypeError(
                f"SimpleGridOpt.GRID_DIM.name parameter must be specified (kwargs[SimpleGridOpt.GRID_DIM.name] = (x, y, z))")
        if not type(kwargs[SimpleGridOpt.GRID_DIM.name]) == type(tuple((0, 0, 0))):
            raise TypeError(
                f"SimpleGridOpt.GRID_DIM.name parameter expected type Tuple[int, int, int], got {type(kwargs[SimpleGridOpt.GRID_DIM.name])}")
        self._dim = Coord(*kwargs[SimpleGridOpt.GRID_DIM.name])
        if SimpleGridOpt.GRID_SIGNALS.name in kwargs:
            signals = kwargs[SimpleGridOpt.GRID_SIGNALS.name]
        else:
            signals = []
        Signal.set_types(signals)
        self.reset()

    def reset(self):
        self._scaffold_grid: Dict[Tuple[int, int, int], Union[Cell, Ecm]] = dict()
        self._medium_grid: Dict[Tuple[int, int, int], List[Signal]] = dict()
        self._reverse_cell_grid: Dict[int, Tuple[int, int, int]] = dict()

    # TODO: temp
    def cell_coords(self):
        return [coord for coord in self._scaffold_grid.keys() if self._scaffold_grid[coord].type == GridObjectType.CELL.name]

    def _reverse_map(self, obj: Cell, coord: Tuple[int, int, int]):
        self._reverse_cell_grid[obj.uuid] = coord

    def find_cell(self, obj_id: int) -> Optional[Cell]:
        if not self._reverse_cell_grid:
            [self._reverse_map(o, c) for c, o in self._scaffold_grid.items() if o.type == GridObjectType.CELL.name]
        try:
            coord = self._reverse_cell_grid[obj_id]
            return self._scaffold_grid[coord]
        except:
            print(f"Grid: Cell with uuid {obj_id} not found in grid.")
            return None

    def set_cell_diff_state(self, _cell_id, _diff_state):
        _cell = self.find_cell(_cell_id)
        if not _cell:
            return False
        self._scaffold_grid[_cell.coord].diff_state = _diff_state
        return True

    @property
    def cells(self):
        return self._get_objects(GridObjectType.CELL.name)

    @property
    def ecm(self):
        return self._get_objects(GridObjectType.ECM.name)

    @property
    def signals(self):
        return self._get_objects(GridObjectType.SIGNAL.name)

    def step(self):
        _signal_grid = deepcopy(self._medium_grid)
        for k, _old_signal_list in _signal_grid.items():
            _new_signal_list = list()
            for _signal in _old_signal_list:
                if _signal.keepalive:  # else automatically discarded
                    _new_coord, _new_drift_dir = self._compute_drift(_signal)
                    if _new_coord:
                        if _new_coord not in self._medium_grid:
                            self._medium_grid[_new_coord] = list()
                        self._medium_grid[_new_coord].append(_signal)
                    else:
                        _new_signal_list.append(_signal)
                    _signal.step(new_coord=_new_coord, new_drift_dir=_new_drift_dir)
                else:
                    del _signal
            self._medium_grid[k] = _new_signal_list

    def _compute_drift(self, s: Signal) -> (Tuple[int, int, int], Tuple[int, int, int]):
        if s.can_drift():
            _new_coord = tuple(map(operator.add, s._coord, s._drift_direction))
            # border_max_reached = [_c >= _d for _c, _d in zip(_new_coord, grid_dim)]
            # border_min_reached = [_c < 0 for _c in _new_coord]
            if self.in_grid(_new_coord):
                return _new_coord, s._drift_direction
            else:
                _new_coord_cropped = self.crop_coord(_new_coord)
                _reverse_dir = [_a != _b for _a, _b in zip(_new_coord_cropped, _new_coord)]
                _new_drift_direction = []
                for _reverse, _old_dir in zip(_reverse_dir, s._drift_direction):
                    if _reverse:
                        _new_drift_direction.append(-1 * _old_dir)
                    else:
                        _new_drift_direction.append(_old_dir)
                return _new_coord_cropped, tuple(_new_drift_direction)
        return None, None

    def in_grid(self, coord: Tuple[int, int, int]):
        return (0 <= coord[0] < self._dim.x and
                0 <= coord[1] < self._dim.y and
                0 <= coord[2] < self._dim.z)

    def crop_coord(self, coord: Tuple[int, int, int]) -> Tuple[int, ...]:
        _new_coord = list()
        for _c, _g in zip(coord, self._dim.tuple):
            if _c >= _g:
                _new_coord.append(_g - 1)
            elif _c < 0:
                _new_coord.append(0)
            else:
                _new_coord.append(_c)
        return tuple(_new_coord)

    def add_where(self, f: Union[str, dict], obj_type: str, **_kwargs) -> List[GridObject]:
        _objs = list()
        if type(f) == str:
            (x, y, z) = (np.array(range(self._dim.x)), np.array(range(self._dim.y)), np.array(range(self._dim.z)))
            var = {'x': x, 'y': y, 'z': z}
            sep = '(>=|<=|==|>|<|=)'  # round brackets prevent discarding separator token
            eqs = f.split(",")
            for eq in eqs:
                left, rel, right = re.split(sep, eq.replace(' ', ''))
                b = eval(eq.replace('x', 'var["x"]').replace('y', 'var["y"]').replace('z', 'var["z"]'))
                var[left] = np.array([_i for _id, _i in enumerate(var[left]) if b[_id]])

            for coord in itertools.product(*(var.values())):
                (x, y, z) = tuple([int(_i) for _i in coord])
                _obj = self.add(obj_type, (x, y, z), **_kwargs)
                _objs.append(_obj)
        elif type(f) == dict:
            coords = Shape.make_shapes(**f)
            # print(coords)
            for coord in coords:
                # print(coord)
                _obj = self.add(obj_type, coord, **_kwargs)
                _objs.append(_obj)
        return _objs

    def neighbor_coords(self, x: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        neighbor_coords: List[Tuple[int, int, int]] = list()
        offset = (-1, 0, 1)
        for i in offset:
            for j in offset:
                for k in offset:
                    neighbor_coords.append(tuple(map(operator.add, x, (i, j, k))))
        return neighbor_coords

    def neighbor_cells(self, x: Union[int, Tuple[int, int, int]]) -> List[int]:
        if type(x) == int:
            coords = self._reverse_cell_grid[x]
        else:
            coords = x
        n_coords = self.neighbor_coords(coords)
        cells = list()
        for neighbor_coord in n_coords:
            if neighbor_coord in self._scaffold_grid:
                _o = self._scaffold_grid[neighbor_coord]
                if _o.type == GridObjectType.CELL:
                    cells.append(_o.uuid)
        return cells

    def available_coords(self, x: Union[int, Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        if type(x) == int:
            coords = self._reverse_cell_grid[x]
        else:
            coords = x
        neighbor_coords = self.neighbor_coords(coords)
        available_coords = list()
        for neighbor_coord in neighbor_coords:
            if self.in_grid(neighbor_coord) and neighbor_coord not in self._scaffold_grid:
                available_coords.append(neighbor_coord)
        return available_coords

    def add_cell_from(self, parent_id: int) -> Optional[GridObject]:
        parent_obj = self.find_cell(parent_id)
        if not parent_obj:
            raise Exception(f"Object id {parent_id} not found")
        parent_coord = parent_obj.coord
        available_coords: List[Tuple[int, int, int]] = list()
        neighbor_coords = self.neighbor_coords(parent_coord)
        for neighbor_coord in neighbor_coords:
            if self.in_grid(neighbor_coord) and neighbor_coord not in self._scaffold_grid:
                available_coords.append(neighbor_coord)
        if not available_coords:
            logging.debug("No available coords for replication")
            return None
        # TODO how about kwargs? maybe copy them from parent...
        child_obj = self.add(parent_obj.type, random.choice(available_coords), **{})
        return child_obj

    def add(self, obj_type: str, coord: Tuple[int, int, int] = None, **_kwargs) -> Optional[GridObject]:
        if obj_type not in GridObjectType.names():
            raise TypeError("obj_type {} not in GridObjectType names ({})".format(obj_type, GridObjectType.names()))
        if coord is None:
            coord = Coord(random.choice(range(self._dim.x)),
                          random.choice(range(self._dim.y)),
                          random.choice(range(self._dim.z)))

        if not self.in_grid(coord):
            print(f"Grid: Object {obj_type} at {coord} outside grid boundaries {self._dim}")
            return None

        if obj_type == GridObjectType.SIGNAL.name:
            if not coord in self._medium_grid:
                self._medium_grid[coord] = list()
            obj = GridObject.new(obj_type, coord, **_kwargs)
            self._medium_grid[coord].append(obj)
            return obj
        else:
            if coord in self._scaffold_grid.keys():
                print(f"Grid: {obj_type} cannot be placed: coord {coord} not empty")
                return None
            elif obj_type == GridObjectType.CELL.name or obj_type == GridObjectType.ECM.name:
                obj = GridObject.new(obj_type, coord, **_kwargs)
                self._scaffold_grid[coord] = obj
                if obj_type == GridObjectType.CELL.name:
                    self._reverse_map(obj, coord)
                return obj

    def remove_at(self, coord: Tuple[int, int, int], obj_type: str, sig_type: str = None, n: int = 1):
        if obj_type == GridObjectType.SIGNAL.name:
            if coord in self._medium_grid:
                if sig_type:
                    if sig_type not in list(Signal.SignalType.names()):
                        raise TypeError(
                            f"sig_type {sig_type} value not in Signal.SignalType names ({Signal.SignalType.names()})")
                    to_remove = n
                    _l = len(self._medium_grid[coord])
                    _list = list()
                    for elem in self._medium_grid[coord]:
                        if to_remove == 0 or elem.signal_type != sig_type:
                            _list.append(elem)
                        else:
                            to_remove -= 1
                    if _list:
                        # print("reinserting", _list)
                        self._medium_grid[coord] = _list
                    else:
                        del self._medium_grid[coord]

                    # _reinsert = list()
                    # for _elem in self._medium_grid[coord]:
                    #     if to_remove <= 0:
                    #         break
                    #     else:
                    #         print(f"{_elem.signal_type} vs {sig_type}")
                    #         if _elem.signal_type == sig_type:
                    #             to_remove -= 1
                    #         else:
                    #             _reinsert.append(_elem)
                    # print("reinsert", _reinsert)
                    # if len(_reinsert) > 0:
                    #     self._medium_grid[coord] = _reinsert
                    #     assert len(self._medium_grid[coord]) > 0
                    # else:
                    #     del self._medium_grid[coord]

                else:
                    # remove all Signals at coord
                    del self._medium_grid[coord]
        else:
            if coord in self._scaffold_grid:
                _l = len(self._scaffold_grid)
                _o = self._scaffold_grid[coord]
                del self._scaffold_grid[coord]
                del self._reverse_cell_grid[_o.uuid]
                assert len(self._scaffold_grid) == _l - 1

    def remove(self, obj: GridObject):
        sig_type = obj.signal_type if type(obj) == Signal else None
        self.remove_at(coord=obj.coord, obj_type=obj.type, sig_type=sig_type, n=1)
        if obj.type == GridObjectType.CELL.name:
            del self._reverse_cell_grid[obj.uuid]

    def remove_cell(self, cell_id: int):
        if cell_id in self._reverse_cell_grid:
            coord = self._reverse_cell_grid[cell_id]
            del self._scaffold_grid[coord]
            del self._reverse_cell_grid[cell_id]

    def _get_objects(self, obj_type: str) -> Optional[List[Union[Cell, Ecm, List[Signal]]]]:
        if obj_type not in GridObjectType.names():
            print(f"Grid: object type {obj_type} not in grid. Accepted object types: {GridObjectType.names()}")
            return None
        if obj_type == GridObjectType.SIGNAL.name:
            return list(self._medium_grid.values())
        else:
            return [x for x in self._scaffold_grid.values() if x.type == obj_type]

    def detached_cells(self) -> List[int]:
        ret = list()
        for cell in self.cells:
            neighbor_coords = [coord for coord in self.neighbor_coords(cell.coord) if
                               coord in self._scaffold_grid.keys()]
            if not neighbor_coords or all(
                    [self._scaffold_grid[coord].type != GridObjectType.ECM.name for coord in neighbor_coords]):
                ret.append(cell.uuid)
        return ret

    def consume_signals(self) -> Dict[int, Dict[str, int]]:
        # 1. intersect signal and cell coords
        # 2. randomly choose a subset
        # 3. for each coord
        #   if at coord there is more than one signal of the same type
        #       3.1 choose low_thresh or high_thresh amount
        #       3.2 output LOW/HIGH -> 1 & 0 / 1 & 1 (e.g. for GF)
        #       3.3 kill those Signal objects
        # 4. associate output to cell@coords
        subset_perc = 1  # TODO settare in seguito la percentuale di cellule che consumeranno segnali, tra quelle che ne hanno a disposizione
        common_coords = list(set(self._medium_grid.keys()) & set(self.cell_coords()))
        # subset_coords = random.sample(common_coords, int(len(common_coords)*subset_perc))
        subset_coords = common_coords
        subset_signals = {_c: self._medium_grid[_c] for _c in subset_coords}
        signal_groups_by_coord = defaultdict() # {(coord1): {'GF': 3, 'Trail': 10}, (coord2): {...}, ...}
        for _c, _sigs in subset_signals.items():
            signal_groups = defaultdict(int)
            for _s in _sigs:
                signal_groups[_s.signal_type] += 1
            signal_groups_by_coord[_c] = signal_groups
        # subset by chosen coords
        signal_groups_by_coord = {k: signal_groups_by_coord[k] for k in signal_groups_by_coord.keys() & subset_coords}
        consumed_signals: Dict[int, Dict[str, int]] = {}
        for _k, _groups in signal_groups_by_coord.items():
            _cell_id = self._scaffold_grid[_k].uuid
            consumed_signals[_cell_id]: Dict[str, int] = {}
            # TODO: nei Signal creati a runtime, inserire anche le regole e poi iterare su quelli!
            # TODO controllare la quantità consumata!
            for _sig_name, _sig_amount in _groups.items():
                # consume at least 1 (random returns a value in [0, 1))
                # _rand_amount = max(1, int(random.random()*_sig_amount))
                _rand_amount = max(1, int(0.7 * _sig_amount)) # consume 70% of the available signal
                consumed_signals[_cell_id][_sig_name] = _rand_amount
                self.remove_at(coord=_k, obj_type="SIGNAL", sig_type=_sig_name, n=_rand_amount)
        return consumed_signals

    # TODO merge into find?
    def get(self, obj_type: str = None, coord: Tuple[int, int, int] = None) -> Optional[Union[GridObject, List[GridObject]]]:
        if coord is None:
            return None
        if obj_type == GridObjectType.SIGNAL.name:
            if coord in self._medium_grid:
                return list(self._medium_grid[coord])
        elif obj_type == GridObjectType.CELL.name or obj_type == GridObjectType.ECM.name:
            if coord in self._scaffold_grid:
                return self._scaffold_grid[coord]

    # TODO reset include_signals to False?
    def draw(self, fig_name="grid.png", include_signals=True, use_offset=True):
        markers = {"ECM": "1", "CELL": "o", "GF": ".", "Trail": "."}
        alpha = {"ECM": 0.02, "CELL": 1, "GF": 0.05, "Trail": 0.05}
        zorder = {"ECM": -1, "CELL": 1, "GF": -1, "Trail": -1}
        colors = {"GF": "limegreen", "Trail": "crimson"}  # {"ECM": "gainsboro", "CELL": "royalblue", "GF": "limegreen", "Trail": "crimson"}
        sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4, "ztick.major.size": 4})
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.view_init(90, 90)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.view_init(0, 0)

        if len(self._scaffold_grid) > 0:
            K = list(self._scaffold_grid.keys())
            (V, COLOR) = zip(*[(elem.type, elem.color) for elem in list(self._scaffold_grid.values())])
            (X, Y, Z) = zip(*K)
            data = pd.DataFrame({"X Value": X, "Y Value": Y, "Z Value": Z, "Category": V, "Color": COLOR})
            # groups = data.groupby("Category")
            groups = data.groupby(["Category", "Color"])
            # print(groups.describe())
            # sleep(3)
            for names, group in groups:
                _name, _color = names
                # print(_name, _color)
                ax1.scatter(group["X Value"], group["Y Value"], group["Z Value"],
                            color=_color, marker=markers[str(_name)], label=_name,
                            alpha=alpha[str(_name)], zorder=zorder[str(_name)])
                # disegno a destra: solo cellule
                if _name != "ECM":
                    ax2.scatter(group["X Value"], group["Y Value"], group["Z Value"],
                                color=_color, marker=markers[str(_name)], label=_name,
                                alpha=alpha[str(_name)], zorder=zorder[str(_name)])

        if include_signals and len(self._medium_grid) > 0:
            _K = list(self._medium_grid.keys())
            V = list(self._medium_grid.values())
            K = list()
            if not use_offset:
                # replicate coords for each signal in list at coords
                K = [_K[_i] for _i, _list in enumerate(V) for _sig in _list]
            else:
                # TODO wrap around zero when negative offset
                # replicate coords for each signal in list at coords, with some offset for better visualization
                for _i, _list in enumerate(V):
                    for _sig in _list:
                        if len(_list) > 1:
                            offset = tuple([round(random.uniform(-1, 1), 2) for _ in range(3)])  # (x, y, z) offset
                            K.append(tuple(map(operator.add, _K[_i], offset)))
                        else:
                            # no offset when only one element at coords
                            K.append(_K[_i])
            try:
                (X, Y, Z) = zip(*K)  # TODO: check
                V = [elem.signal_type for l in V for elem in l]
                data = pd.DataFrame({"X Value": X, "Y Value": Y, "Z Value": Z, "Category": V})
                groups = data.groupby("Category")
                for name, group in groups:
                    ax1.scatter(group["X Value"], group["Y Value"], group["Z Value"],
                                color=colors[str(name)], marker=markers[str(name)], label=name,
                                alpha=alpha[str(name)],  zorder=zorder[str(name)])
                    ax2.scatter(group["X Value"], group["Y Value"], group["Z Value"],
                                color=colors[str(name)], marker=markers[str(name)], label=name,
                                alpha=alpha[str(name)],  zorder=zorder[str(name)])
            except Exception as e:
                # N.B. eccezione solo nell'ultimo plot?
                print("Grid: Skipping signals because... whatever. K???")

        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        plt.xlim(0, self._dim.x)
        plt.ylim(0, self._dim.y)
        ax1.set_zlim(0, self._dim.z)
        ax2.set_zlim(0, self._dim.z)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        plt.savefig(fig_name)
        plt.close()


class GridSimulator(ISimulator):
    _grid: Any
    _draw_interval: int
    _t: int

    class Event(AdvEnum):
        # TODO make these enumerations public and define ony in one place (e.g. death and replication in bool sim only
        # for reading input events
        DEATH = 0
        REPLICATION = 1
        # for output events
        SPAWNED = 2
        REMOVED = 3
        ECM_LOSS = 4
        CELLDENSITY_HIGH = 5
        CELLDENSITY_LOW = 6
        CONSUMED_SIGNALS = 7

    def __init__(self):  # , InputOutput.PROTOCOL
        super().__init__([InputOutput.CONFIG], [], [InputOutput.GRID_EVENTS], name="GridSimulator")
        self._output_path = None

    @Pyro4.expose
    def reset(self):
        self._grid.reset()
        self._t = 0

    def _setup(self, **kwargs) -> Optional[Dict]:
        print(f"{self.__class__.__name__} setup...")
        if not all([k in kwargs.keys() for k in SimpleGridOpt.names()]):
            raise ValueError(
                f"Module setup requires parameters {SimpleGridOpt.names()} in {InputOutput.GRID_CONFIG} input.")
        kwargs[SimpleGridOpt.GRID_DIM.name] = tuple(kwargs[SimpleGridOpt.GRID_DIM.name])
        self._grid = Grid(**kwargs)
        self._draw_interval = int(kwargs[SimpleGridOpt.GRID_DRAW_INTERVAL.name]) if kwargs[SimpleGridOpt.GRID_DRAW_INTERVAL.name] else -1
        self._t = 0
        _objs, _new_cells = self._process_grid_add(**kwargs)
        return _new_cells

    def _process_grid_add(self, **kwargs) -> (List[GridObject], Dict[int, Optional[int]]):
        _objs: List[GridObject] = list()
        if SimpleGridOpt.GRID_ADD.name in kwargs:
            for elem in kwargs[SimpleGridOpt.GRID_ADD.name]:
                if len(elem) > 2:
                    obj_type, coords, opt = elem
                else:
                    obj_type, coords = elem
                    opt = None
                _obj = self._grid.add(obj_type, coords, **opt) if opt else self._grid.add(obj_type, coords)
                _objs += [_obj]
            del kwargs[SimpleGridOpt.GRID_ADD.name]

        if SimpleGridOpt.GRID_ADD_WHERE.name in kwargs:
            for elem in kwargs[SimpleGridOpt.GRID_ADD_WHERE.name]:
                if len(elem) > 2:
                    cond, obj_type, opt = elem
                else:
                    cond, obj_type = elem
                    opt = None
                _objs += self._grid.add_where(cond, obj_type, **opt) if opt else self._grid.add_where(cond, obj_type)
            del kwargs[SimpleGridOpt.GRID_ADD_WHERE.name]

        # k:v map to child_cell:parent_cell; parent_cell < 0 means no parent
        _new_cells: Dict[int, Optional[int]] = dict()
        for _obj in _objs:
            if type(_obj) == Cell:  # skip Ecm
                _new_cells[_obj.uuid] = None
        return _objs, _new_cells

    def add_model(self):
        raise Exception(
            f"Class {self.__class__.__name__} simulates only one instance of {Grid.__class__.__name__} class.")

    @Pyro4.expose
    def data(self):
        try:
            res = self._grid.cell_coords()
        except:
            res = list()
        return res

    def remove_model(self):
        pass

    @Pyro4.expose
    def step(self, *args, **kwargs) -> (tuple, dict):
        super(self.__class__, self).step(**kwargs)
        if not kwargs or not all([_in.name in kwargs.keys() for _in in self.input_list]):
            raise ValueError(f"The module requires this input: {self.input_list}.")

        if InputOutput.GRID_CONFIG.name in kwargs[InputOutput.CONFIG.name]:
        # if self._grid is None:
            if kwargs[InputOutput.CONFIG.name][InputOutput.OUTPUT_PATH.name]:
                self._output_path = os.path.join(kwargs[InputOutput.CONFIG.name][InputOutput.OUTPUT_PATH.name], self.name)
            else:
                self._output_path = os.path.join(os.path.dirname(__file__), self.name)
            if not os.path.isdir(self._output_path):
                os.makedirs(self._output_path, exist_ok=True)

            new_cells = self._setup(**kwargs[InputOutput.CONFIG.name][InputOutput.GRID_CONFIG.name])
            del kwargs[InputOutput.CONFIG.name][InputOutput.GRID_CONFIG.name]  # config done, remove info
            kwargs[InputOutput.GRID_EVENTS.name] = {}
            for _k in GridSimulator.Event.names():
                kwargs[InputOutput.GRID_EVENTS.name][_k] = list()
            kwargs[InputOutput.GRID_EVENTS.name][GridSimulator.Event.SPAWNED.name] = new_cells

        #else:
        logging.info(f"{self.__class__.__name__} stepping...")
        # process incoming events
        # (simple_grid is the master simulator, so incoming events come first because they were generated during the previous step)
        if InputOutput.LIFECYCLE_EVENTS.name in kwargs:
            # logging.info("GridSimulator: processing Lifecycle events")
            print(f"@{self._t} {self.name}: processing Lifecycle events")
            lifecycle_events = kwargs[InputOutput.LIFECYCLE_EVENTS.name]
            if not kwargs[InputOutput.GRID_EVENTS.name]:
                kwargs[InputOutput.GRID_EVENTS.name] = {}

            if GridSimulator.Event.DEATH.name in lifecycle_events.keys():
                # logging.info("GridSimulator: processing dead cells")
                print(f"@{self._t} {self.name}: processing {len(lifecycle_events['DEATH'])} dead cells")
                removing_cells = list()
                celldensity_low = list()
                for _cell_id in lifecycle_events[GridSimulator.Event.DEATH.name]:
                    removing_cells.append(_cell_id)
                for _cell_id in removing_cells:
                    _neighbors = list(set(self._grid.neighbor_cells(_cell_id)) - set(removing_cells))
                    celldensity_low.extend(_neighbors)
                [self._grid.remove_cell(_cell_id) for _cell_id in removing_cells]
                kwargs[InputOutput.GRID_EVENTS.name][GridSimulator.Event.REMOVED.name] = removing_cells
                kwargs[InputOutput.GRID_EVENTS.name][GridSimulator.Event.CELLDENSITY_LOW.name] = celldensity_low
            if GridSimulator.Event.REPLICATION.name in lifecycle_events.keys():
                # logging.info("GridSimulator: processing replicating cells")
                print(f"@{self._t} {self.name}: processing replicating cells")
                spawned_cells: Dict[int, int] = dict()
                celldensity_high = list()
                for _parent_id in lifecycle_events[GridSimulator.Event.REPLICATION.name]:
                    _child_obj = self._grid.add_cell_from(_parent_id)
                    if _child_obj:
                        spawned_cells[_child_obj.uuid] = _parent_id
                    else:
                        celldensity_high.append(_parent_id)

                # N.B. ...SPAWNED is not empty here, because Lifecycle simulator does not delete that key yet
                # (might be useful to some other simulator in the future)
                kwargs[InputOutput.GRID_EVENTS.name][GridSimulator.Event.SPAWNED.name] = spawned_cells
                kwargs[InputOutput.GRID_EVENTS.name][GridSimulator.Event.CELLDENSITY_HIGH.name] = celldensity_high

        if InputOutput.DIFFERENTIATION_EVENTS.name in kwargs:
            diff_events = kwargs[InputOutput.DIFFERENTIATION_EVENTS.name]
            # logging.info("GridSimulator: processing differentiating cells")
            print(f"@{self._t} {self.name}: processing differentiating cells")
            for _diff_state, _cell_ids in diff_events.items():
                # skip dead cells
                if _diff_state == DiffStates.APOPTOSIS.name:
                    continue
                # update cell color based on its differentiation state
                for _cell_id in _cell_ids:
                    # self._grid.set_cell_color(_cell_id, eval(DiffStateColors.__name__ + "." + _diff_state + ".value"))
                    self._grid.set_cell_diff_state(_cell_id, _diff_state)
                # if _cell_ids:
                #     print(f"{_diff_state}: {_cell_ids}")
                # [self._grid]
                pass

        # process protocol
        if InputOutput.DEMIURGOS_EVENTS.name in kwargs:
            demiurgos_events = kwargs[InputOutput.DEMIURGOS_EVENTS.name]
            print(f"@{self._t} {self.name}: processing Demiurgos events")
            _objs, _cells = self._process_grid_add(**demiurgos_events)

        # process local events (as of now, only ECM loss and consumed SIGNALs)
        # grid steps first, calculating signal drift
        logging.info(f"@{self._t} {self.name}: performing local step")
        self._grid.step()
        detached_cells = self._grid.detached_cells()
        consumed_signals = self._grid.consume_signals()
        kwargs[InputOutput.GRID_EVENTS.name][GridSimulator.Event.ECM_LOSS.name] = detached_cells
        kwargs[InputOutput.GRID_EVENTS.name][GridSimulator.Event.CONSUMED_SIGNALS.name] = consumed_signals

        # check if there are no cells in simulation
        if len(self._grid.cell_coords()) <= 0:
            print(f"@{self._t} {self.name}: 0 cells, stopping simulation")
            kwargs[InputOutput.STOP_SIMULATION.name] = True

        # TODO draw if flag...
        if self._t % self._draw_interval == 0:
            self.draw(fig_name=os.path.join(self._output_path, f"{self._t}_grid.png"))

        self._t += 1

        return args, kwargs

    # TODO check path! (in Pipeline)
    def draw(self, *args, **kwargs):
        # if self._t == 0 or self._draw_interval <= 0 or (self._t + 1) % self._draw_interval == 0:
        self._grid.draw(*args, **kwargs)


# def main():
#     x = y = z = 16
#     signals = ["AAA", "BBB", "CCC"]
#     g = Grid(**{SimpleGridOpt.GRID_DIM.name: (x, y, z), SimpleGridOpt.GRID_SIGNALS.name: signals})
#     # c = Cell((11, 11, 11), radius=1)
#     g.add(GridObjectType.CELL.name, (11, 11, 11), **{Cell.Opt.RADIUS.name: 1})
#     g.add(GridObjectType.CELL.name)
#     g.add(GridObjectType.ECM.name, (1, 2, 3))
#     g.add_where('z<=1', GridObjectType.ECM.name)
#     g.add_where('x>=0, x<=2, y>=10, y<=15, z>3, z<=6', GridObjectType.CELL.name)
#     g.add_where('x>=4, x<=6, y>=10, y<=15, z>3, z<=6', GridObjectType.CELL.name)
#     g.add_where('x>=2, x<=4, y>=0, y<=10, z>3, z<=6', GridObjectType.CELL.name)
#     for _ in range(10):
#         coord = (random.choice(range(x)), random.choice(range(y)), random.choice(range(z)))
#         g.add(GridObjectType.SIGNAL.name, coord)
#     c = g.get(GridObjectType.CELL.name, (11, 11, 11))
#     g.remove(c)
#
#     g.draw(include_signals=True)
#
#
# def main2():
#     x = y = z = 16
#     signals = ["AAA", "BBB", "CCC"]
#     g = Grid(**{SimpleGridOpt.GRID_DIM.name: (x, y, z), SimpleGridOpt.GRID_SIGNALS.name: signals})
#     g.add(GridObjectType.CELL.name, (0, 0, 0))
#     c1 = g.add(GridObjectType.CELL.name, (1, 1, 1))
#     g.add(GridObjectType.CELL.name, (5, 6, 7))
#     g.add(GridObjectType.CELL.name, (8, 2, 1))
#     g.add(GridObjectType.ECM.name, (2, 2, 2))
#     s = list()
#     for i in range(3, 6):
#         _s = g.add(GridObjectType.SIGNAL.name, (i, i, i))
#         s.append(_s)
#     kwargs = {Signal.Opt.TYPE.name: Signal.SignalType.AAA.name}
#     for _ in range(5):
#         g.add(GridObjectType.SIGNAL.name, (7, 7, 0), **kwargs)
#     g.add(GridObjectType.SIGNAL.name, (7, 7, 0), **{Signal.Opt.TYPE.name: Signal.SignalType.BBB.name})
#     g.draw(fig_name="aaa.png", include_signals=True)
#     print(s)
#     g.remove(s[0])
#     g.remove_at((7, 7, 0), GridObjectType.SIGNAL.name, sig_type=Signal.SignalType.AAA.name, n=2)
#     g.draw(fig_name="bbb.png", include_signals=True)
#     g.remove_at(coord=(7, 7, 7), obj_type=GridObjectType.SIGNAL.name, sig_type=Signal.SignalType.BBB.name)
#     g.add(GridObjectType.SIGNAL.name, (7, 7, 7))  # , {Signal.Opt.TYPE.name: Signal.SignalType.BBB.name}
#     g.remove_at(coord=(7, 7, 7), obj_type=GridObjectType.SIGNAL.name, sig_type=Signal.SignalType.BBB.name)
#     g.remove_at(coord=(7, 7, 7), obj_type=GridObjectType.SIGNAL.name, sig_type=Signal.SignalType.AAA.name, n=1)
#     g.remove_at(coord=(7, 7, 7), obj_type=GridObjectType.SIGNAL.name, sig_type=Signal.SignalType.AAA.name)
#     g.remove(c1)
#     g.draw(fig_name="ccc.png", include_signals=True)
#
#
def main_prova():
    path = Path("coherence/experiments/test")
    config = {InputOutput.PATH.name: path,
              InputOutput.DEBUG_PATH.name: '.',
              InputOutput.CONFIG.name: {
                  InputOutput.GRID_CONFIG.name:
                      {"GRID_DIM": (50, 50, 20),
                       "GRID_TIMESCALE": 1,
                       "GRID_SIGNALS": ['GF', 'Trail'],
                       "GRID_ADD": [
                           ["SIGNAL", (0, 0, 0), {'TYPE': 'GF', 'DRIFT_DIRECTION': (10, 10, 3)}]
                       ],
                       # "GRID_ADD": [
                       #     ["CELL", (20, 20, 6)],
                       #     ["CELL", (5, 6, 7), {'RADIUS': 1}],
                       #     ["CELL", (15, 15, 15)],
                       #     ["CELL", (18, 18, 5)]
                       # ],
                       # "GRID_ADD_WHERE": [
                       #     [{"CUBOID": {"WIDTH": 40, "DEPTH": 40, "HEIGHT": 3, "ORIGIN": [0, 0, 0]}}, "ECM"],
                       #     [{"TORUS": {"CENTER": [10, 10, 10], "MAJOR_RADIUS": 8, "MINOR_RADIUS": 2}}, "ECM"]
                       # ],
                       # "GRID_ADD_WHERE": [
                       #     ['x<50, y<50, z<3', "ECM", {'STIFF': True}],
                       #     ['x<10, y<10, z>17', "CELL"]
                       # ],
                       "GRID_ADD_WHERE": [],
                       "GRID_DRAW_INTERVAL": 0
                       }
              }
              }

    g = GridSimulator()
    _, config = g.step(**config)
    # print(config)
    new_cells = config[InputOutput.GRID_EVENTS.name][GridSimulator.Event.SPAWNED.name]
    print(f"new_cells {new_cells}")
    # let x% of new_cells replicate; just for testing
    for _ in range(5):
        config[InputOutput.LIFECYCLE_EVENTS.name] = {
            GridSimulator.Event.REPLICATION.name: [random.choice(list(new_cells.keys())) for _ in
                                                   range(int(len(new_cells) * 0.2))]}
        _, config = g.step(**config)
        # print(config)
    # let x% of new_cells die; just for fun
    g.draw()

    for _ in range(5):
        config[InputOutput.LIFECYCLE_EVENTS.name] = {
            GridSimulator.Event.DEATH.name: [random.choice(list(new_cells.keys())) for _ in
                                             range(int(len(new_cells) * 0.3))]}
        # g._grid.add("SIGNAL", (20, 20, 12), **{'TYPE': 'GF', 'DRIFT_DIRECTION': (0, 0, 3)})
        _, config = g.step(**config)
        g.draw()
        # print(config)


if __name__ == "__main__":
    #main_prova()
    import sys

    print(sys.argv)
    assert (len(sys.argv) > 2)
    host = sys.argv[1]
    port = int(sys.argv[2])
    print(f"port {port}")
    kwargs = {"host": host, "port": port}
    GridSimulator.serve(**kwargs)
