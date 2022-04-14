#TODO leggere i segnali da GRID_CONFIG -> GRID_SIGNALS, come fatto in GridSimulator
from cosimo.simulator import *
from cosimo.utils import AdvEnum

from library.common.utils import InputOutput


class Signal(AdvEnum):
    GF = 0
    Trail = 1
    # output only
    GF_High = -1


class SignalTranslator(ISimulator):
    class Event(AdvEnum):
        SPAWNED = 0
        REMOVED = 1
        CONSUMED_SIGNALS = 2

    _t: int
    _last_fed: dict
    _feeding_threshold = 20
    _protocol_step: int
    _gf_high_threshold = 1  # GF is set as High when consuming *strictly* more (>) than this amount
    _default_protocol = {Signal.GF.name: 0, Signal.GF_High.name: 0, Signal.Trail.name: 0}

    def __init__(self):
        super().__init__([], [], [InputOutput.SIGNAL_TRANSLATOR_EVENTS], "SignalTranslator")
        self.reset()

    @Pyro4.expose
    def reset(self):
        self._t = 0
        self._last_fed = dict()

    @Pyro4.expose
    def step(self, *args, **kwargs) -> (tuple, dict):
        super(self.__class__, self).step(**kwargs)
        if not kwargs or not all([_in.name in kwargs.keys() for _in in self.input_list]):
            raise ValueError(f"The module requires this input: {self.input_list}.")

        if kwargs[InputOutput.GRID_EVENTS.name]:
            grid_events = kwargs[InputOutput.GRID_EVENTS.name]

            if grid_events[SignalTranslator.Event.SPAWNED.name]:
                for cell_id in grid_events[SignalTranslator.Event.SPAWNED.name].keys():
                    self._last_fed[cell_id] = -1 #note: it becomes zero when processing the presence/absence from SIGNALS
            if grid_events[SignalTranslator.Event.REMOVED.name]:
                for cell_id in grid_events[SignalTranslator.Event.REMOVED.name]:
                    del self._last_fed[cell_id]

            if not grid_events[SignalTranslator.Event.CONSUMED_SIGNALS.name]:
                out_signals = self._set_default_protocol(self._last_fed.keys())
                kwargs[InputOutput.SIGNAL_TRANSLATOR_EVENTS.name] = out_signals
            else:
                consumed_signals = grid_events[SignalTranslator.Event.CONSUMED_SIGNALS.name]
                not_fed = list(set(self._last_fed.keys()) - set(consumed_signals.keys()))
                out_signals = self._set_default_protocol(not_fed)
                for cell_id, in_signals in consumed_signals.items():
                    self._last_fed[cell_id] = 0
                    out_signals[cell_id] = dict()
                    out_signals[cell_id][Signal.Trail.name] = 1 if Signal.Trail.name in in_signals else 0
                    if Signal.GF.name in in_signals:
                        out_signals[cell_id][Signal.GF.name] = 1
                        out_signals[cell_id][Signal.GF_High.name] = 1 if in_signals[Signal.GF.name] > self._gf_high_threshold else 0
                del grid_events[SignalTranslator.Event.CONSUMED_SIGNALS.name]
                kwargs[InputOutput.SIGNAL_TRANSLATOR_EVENTS.name] = out_signals

        if kwargs[InputOutput.SIGNAL_TRANSLATOR_EVENTS.name]:
            pass
            #print(f"{self.__class__.name}@{str(self._t).zfill(4)}:")
            #print(kwargs[InputOutput.SIGNAL_TRANSLATOR_EVENTS.name])
        self._t += 1
        return args, kwargs

    def _set_default_protocol(self, ids) -> dict:
        ret = dict()
        for cell_id in ids:
            self._last_fed[cell_id] += 1
            if self._last_fed[cell_id] > self._feeding_threshold:
                self._last_fed[cell_id] = 0
                ret[cell_id] = self._default_protocol
        return ret

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
    SignalTranslator.serve(**kwargs)
