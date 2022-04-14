from cosimo.simulator import *

from library.common.utils import AdvEnum, InputOutput


class Collector(ISimulator):
    _t: int
    _stats: dict

    class Event(AdvEnum):
        SPAWNED = 0
        REMOVED = 1

    def __init__(self):
        super().__init__([InputOutput.GRID_EVENTS, InputOutput.LIFECYCLE_EVENTS], [], [], "Collector")
        self.reset()

    @Pyro4.expose
    def reset(self):
        self._t = 0
        self._stats = dict()

    @Pyro4.expose
    def step(self, *args, **kwargs) -> (tuple, dict):
        super(self.__class__, self).step(**kwargs)
        if not kwargs or not all([_in.name in kwargs.keys() for _in in self.input_list]):
            raise ValueError(f"The module requires this input: {self.input_list}.")

        for event in Collector.Event.names():
            if event not in self._stats.keys():
                self._stats[event] = list()
            self._stats[event].append(len(kwargs[InputOutput.GRID_EVENTS.name][event]) if event in kwargs[InputOutput.GRID_EVENTS.name] else 0)

        self._t += 1

        return args, kwargs

    @Pyro4.expose
    def data(self):
        # logging.debug(f"Collector stats\n{pd.DataFrame.from_dict(data=self._stats)}")
        return self._stats

    def add_model(self, *args, **kwargs):
        raise NotImplementedError

    def remove_model(self, *args, **kwargs):
        # TODO
        pass


if __name__ == "__main__":
    import sys

    print(sys.argv)
    assert (len(sys.argv) > 2)
    host = sys.argv[1]
    port = int(sys.argv[2])
    print(f"port {port}")
    kwargs = {"host": host, "port": port}
    Collector.serve(**kwargs)
