[![DOI](https://zenodo.org/badge/481348517.svg)](https://zenodo.org/badge/latestdoi/481348517)

# How to define an experiment
When you define a new experiment to be run in coherence, you must provide four configuration files and store them in a separate directory `$EXPERIMENT_NAME`.

One configuration file is required by coherence itself and contains high-level parameters for the simulation and DSE engines. [Three additional configuration files](#μgp3-setup-parameters) are required by μgp3. 

The examples below are taken from the configuration files of [the "half" experiment](./half), where the target is a rectangle covering half the culturing area. Additional examples are [a stripe](./stripe) and [a circle](./circle).

# Coherence setup parameters
The parameters described in the following are listed in the `./coherence_conf.py` file.

## Simulation engine
### Simulation pipeline
Coherence is based on [CoSimo](https://github.com/leonardogian/CoSimo), a prototypal general-purpose compositor for loosely-coupled co-simulations.
A `Simulation` is automatically set up given a list of simulators inside the `META` dictionary. Lines 4-22 specify the `Simulators` to be instantiated locally, or instances of `Simulators` to connect to that running on remote machines. The `Simulators` will be executed in the same order as they are listed.

For instance, `Demiurgos` is the name of an instance of `Demiurgos` class (a `Simulator` subclass written in python and included in `coherence/library`) to be run locally.
Alternatively, you can specify the address of a `Demiurgos` instance running on a remote machine.

> It is your responsibility to execute the remote `Simulators` before running Coherence.

> More in-depth information about `Simulator` and `Simulation` classes are available in the [`CoSimo` repository](https://github.com/leonardogian/CoSimo). 

```python
META = {
    'Demiurgos': {
        'python': 'library.demiurgos:Demiurgos',
        # 'remote': 'user@machine.domain:4242'
    },
    'GridSimulator': {
        'python': 'library.space.simple_grid:GridSimulator',
    },
    ...
    'MySimulatorName': {
        'python': 'path.to.module:Class',        
    }
}

```

### Simulators configuration
Lines 23-31 and 32-44 specify the configurations for two of the chained `Simulators`, [`SpatialGrid`](../library/space/simple_grid.py) and [`LifecycleSimulator`](../library/lifecycle/models/bool_lifecycle.py), respectively.

```python
GRID_CONFIG = {"GRID_DIM": (200, 200, 200),
               "GRID_TIMESCALE": 1,
               "GRID_SIGNALS": ['GF', 'Trail'],
               "GRID_ADD_WHERE": [
                   [{"CUBOID": {"WIDTH": 200, "DEPTH": 200, "HEIGHT": 3, "ORIGIN": [0, 0, 0]}}, "ECM"],
               ],
               "MIN_Z": 3,
               "GRID_DRAW_INTERVAL": 20
               }
```

* `GRID_DIM` specifies the size of the 3D culturing space
* `GRID_TIMESCALE` rescales `GridSimulator` simulation step to the whole simulation. In this case, `GridSimulator` is executed at each simulation step.
* `GRID_SIGNALS` are inputs provided in the culturing protocol, used by `GridSimulator` to subclass its `Signal` class.
* `GRID_ADD_WHERE` is a compact way to initialize the grid and uses constructs from the [`geometry` utility module](../library/common/geometry.py). In this case, it is filled with a 200x200x3 layer made of extracellular matrix (`ECM`) objects.
* `MIN_Z` parameter is currently not being used.
* `GRID_DRAW_INTERVAL` is the number of steps between each of the subsequent *snapshot* images showing the content of the 3D culturing space.

```python
LIFECYCLE_CONFIG = {"LIFECYCLE_INIT_STATE": REGAN_INIT_STATES.PROLIFERATING_1.value,
                    "LIFECYCLE_TIMESCALE": 1,
                    "LIFECYCLE_DRAW_INTERVAL": 1,
                    "LIFECYCLE_MANAGE_DEATH": True,
                    }
```
* `LIFECYCLE_INIT_STATE` specifies the cell cycle state to initialize new cells in, according to [Syzek et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006402)
* `LIFECYCLE_MANAGE_DEATH` must always be set to `True` in this version of Coherence.

### Simulation configuration
The `Simulation` itself requires some parameters to be specified:

```python
SIM_CONFIG = {
    "CONFIG": {
        "SIM_STEPS": 1500,
        "GRID_CONFIG": GRID_CONFIG,
        "LIFECYCLE_CONFIG": LIFECYCLE_CONFIG,
    }
}
```

* `SIM_STEPS` is the maximum number of simulation steps to be run. A simulation can end earlier if no cell survived.
* `<simulator>_CONFIG` entries receive each a `Simulator` configuration. In this case, only `GridSimulator` and `LifecycleSimulator` require a configuration dictionary.

## DSE engine

The DSE engine configuration (lines 46-53 in [coherence_conf.py](./half/coherence_conf.py)) is very straightforward:

```python
HALF = {"DESCR": {"CUBOID": {"width": 200, "DEPTH": 100, "HEIGHT": 1, "ORIGIN": (0, 0, 3)}},
        "BOUNDING_BOX": ((0, 200), (0, 100), (3, 4))}
```

The lines above specify the geometry of the desired target product, using primitives from our [geometry library](../library/common/geometry.py).

In this example, the target is a `200x100x1` parallelepiped placed at `z=3`, on top of the ECM defined above.

The following lines set the target to the one defined above, and define a `PROTOCOL_STEP`, that is, how many simulation steps are executed between two subsequent protocol instructions.
The `MASK` entry is a binary 2D array used for the computation of the fitness values.

```python
UGP_CONFIG = {
    "PROTOCOL_STEP": 5,
    "TARGET": HALF,
    "MASK": geometry.get_mask(GRID_CONFIG["GRID_DIM"], HALF["DESCR"])
}
```

# μgp3 setup parameters
μgp3 configuration is spread in three separate xml files - `coherence.constraints.xml`, `coherence.population.settings.xml`, and `ugp3.settings.xml`.
Here a short description of some parameters is provided. For a detailed explanation you can refer to the [μgp3 textbook](https://link.springer.com/book/10.1007/978-0-387-09426-7).

## coherence.constraints.xml
This file defines the structure of an individual (i.e., a protocol), as being composed by a *placing* section (lines 22-45) followed by a *protocol* section (lines 46-90).
The first one contains 1-15 3D coordinates for placing cells at the beginning of the biofabrication process. The latter lists 300 subsections (1500 simulation steps divided by the protocol step, 5), each containing 0-50 macros among `gf_high`, `gf_low`, and `trail`.

## coherence.population.settings.xml
This file defines a `MultiObjective` population, with a 2-value fitness (lines 5 and 18). The initial population, generated randomly by μgp3, is composed of `ν=10` individuals and has maximum size `μ=10`. At each evolutionary step, `γ=10` genetic operators are randomly chosen among the ones specified in lines 80-84 and applied. The `eliteSize` parameter specifies that no matter how better the new individuals are, the `1` best individual from the previous generation survives to be part of the next. 

Line 74 is used to set the evaluator program - [`coherence.py`](../coherence.py) - used by μgp3 to rank each individual. The evaluator program must write the 2-value fitness in a `fitness.out` file (line 76).

μgp3 does not provide support parallel evaluation, which has to be provided by the evaluator program. Therefore, if the `concurrentEvaluations` option (line 72) is set to a value greater than 1, then the evaluator program must be able to output the 2-value fitness arrays as ugp3 expects it: one per line, in the same order as the individuals are generated. `coherence.py` already supports this feature (i.e., you can set `concurrentEvaluations` to values greater than 1).

## ugp3.settings.xml
* randomSeed (line 5) - this option is set to a known value for reproducibility
* populations (lines 7-9) - only one population has been defined; you can add more, together with their configuration file
* statisticsPathName (line 11) - name of the file for μgp3 to store statistics data