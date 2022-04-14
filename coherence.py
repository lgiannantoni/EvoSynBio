import ast
import logging
import os
import sys
from copy import deepcopy
from time import sleep
from typing import Dict, List

from cosimo.simulation import Simulation

# spostare in coherence_conf.py?
DEBUG = False
SLEEP = 100  # 1000


class Coherence(Simulation):
    sim_config = None
    ugp_config = None
    protocol_count = None
    main_output_path = "results"
    curr_experiment_path = main_output_path  # for compatibility with simulators using ugp4, that need to read this
    curr_protocol_path = None

    def __new__(cls, *args, **kwargs):
        super().__new__(cls, *args, **kwargs)
        if not os.path.isdir(Coherence.main_output_path):
            os.mkdir(Coherence.main_output_path)

    @classmethod
    def optimize_protocol(cls, *args, **kwargs):
        cls.sim_config = kwargs["SIM_CONFIG"]
        cls.ugp_config = kwargs["UGP_CONFIG"]

    # TODO non si dovrebbe accedere così alla pipeline...
    @classmethod
    def results(cls):
        # TODO check setup done
        res = dict()
        for sim in cls._simulation_pipeline._pipe:
            res[sim.name] = sim.data()
        return res

    @staticmethod
    def _write_individual(individual, fitness):
        _placing_path = os.path.join(Coherence.curr_protocol_path, "placing.txt")
        _protocol_path = os.path.join(Coherence.curr_protocol_path, "protocol.txt")
        _fitness_path = os.path.join(Coherence.curr_experiment_path, "fitness.txt")
        cells, protocol = Coherence._get_placing_protocol(individual)
        with open(_placing_path, 'w') as fout:
            print(f"{cells}\n", file=fout)
        with open(_protocol_path, 'w') as fout:
            for _i, _p in protocol.items():
                print(f"{_i}: {_p}\n", file=fout)
        with open(_fitness_path, 'a') as fout:
            print(str(Coherence.protocol_count) + '\t' + '\t'.join(str(_f) for _f in fitness), file=fout)

    @staticmethod
    def _get_placing_protocol(ind: str) -> (list, dict):
        s_placing, s_protocol = str(ind).lstrip("cell placing\n").rstrip("\n").replace('\'', '').split("protocol")
        # process placing
        placing = [ast.literal_eval(coord) for coord in s_placing.rstrip("\n").split("\n")]
        it = [_tk for _tk in s_protocol.split("\n") if _tk]
        protocol_dict: Dict[int, List[str]] = dict()
        # step = -1
        step = -Coherence.ugp_config["PROTOCOL_STEP"]
        for _i, _tk in enumerate(it):
            if _tk == "*":
                step += Coherence.ugp_config["PROTOCOL_STEP"]
                protocol_dict[step] = list()
            else:
                protocol_dict[step].append(_tk)
        return placing, protocol_dict

    @staticmethod
    def _eval_protocol(individual_file):
        if DEBUG:
            import random
            # n.b. numero di valori di fitness e range dipendono da maximumFitness in coherence.population.settings.xml
            return [random.uniform(0.0, 100.0), random.uniform(0.0, 100.0)]

        fitness = []
        with open(individual_file, "r") as fin:
            try:
                individual = fin.read()

                logging.info("Evaluating protocol")
                placing, protocol = Coherence._get_placing_protocol(individual)
                _sim_config = deepcopy(Coherence.sim_config)
                Coherence.curr_protocol_path = os.path.join(Coherence.curr_experiment_path,
                                                            "protocol_" + str(Coherence.protocol_count))
                os.mkdir(Coherence.curr_protocol_path)
                _sim_config["CONFIG"]["OUTPUT_PATH"] = str(Coherence.curr_protocol_path)
                _sim_config["CONFIG"]["UGP_CONFIG"] = Coherence.ugp_config
                _sim_config["PROTOCOL"] = protocol
                _sim_config["PLACING"] = placing
                Coherence.start(**_sim_config)
                data = Coherence.results()
                fitness = list()
                target_area = sum(Coherence.ugp_config["MASK"].values())
                expected_cells = [Coherence.ugp_config["MASK"][key] == 1 for key in
                                  data["GridSimulator"] & Coherence.ugp_config["MASK"].keys()].count(True)
                unexpected_cells = [Coherence.ugp_config["MASK"][key] == 0 for key in
                                    data["GridSimulator"] & Coherence.ugp_config["MASK"].keys()].count(True)
                result_area = expected_cells + unexpected_cells
                x, y, _ = Coherence.sim_config["CONFIG"]["GRID_CONFIG"]["GRID_DIM"]
                total_area = x * y
                round_to_decimals = 1

                coverage = round(100.0 * expected_cells / target_area, round_to_decimals)
                precision = round(100.0 * expected_cells / (1 + expected_cells + unexpected_cells), round_to_decimals)
                fitness = [precision, coverage]  # , delta_growth]
                logging.info(f"--------- fitness = {fitness}")
                Coherence._write_individual(individual, fitness)
                Coherence.reset()

            except Exception as e:
                raise e
        return fitness


def main(proc_num, individual, return_dict):
    sys.path.append(os.getcwd())
    from coherence_conf import META, UGP_CONFIG, SIM_CONFIG

    Coherence(**META, shutdown=True)
    try:
        Coherence.optimize_protocol(**{"UGP_CONFIG": UGP_CONFIG, "SIM_CONFIG": SIM_CONFIG})
    except Exception as e:
        raise e
    try:
        Coherence.protocol_count = individual.split('.')[0].split(sep='_')[1].replace('*', 'x')  #get individual (protocol) "name" from e.g. individual_A*.in
        fitness = Coherence._eval_protocol(individual)
        return_dict[proc_num] = fitness
    except Exception as e:
        print(e)
    finally:
        pass


if __name__ == "__main__":
    import multiprocessing

    _individuals = sys.argv[1:]
    n_cpu = os.cpu_count()
    assert len(_individuals) <= n_cpu, f"Number of individuals ({len(_individuals)}) greater than number of CPUs ({n_cpu})! " \
                                       f"Consider modifying concurrentEvaluations parameter in coherence.population.settings.xml."

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for _proc_num, _individual in enumerate(_individuals):
        p = multiprocessing.Process(target=main, args=(_proc_num, _individual, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    return_dict = dict(sorted(return_dict.items()))

    with open("fitness.out", "w") as fout:
        for _proc_id, _fitness in return_dict.items():
            [fout.write('%s ' % _f) for _f in _fitness]
            fout.write('\n')
    if DEBUG:
        sleep(SLEEP)
