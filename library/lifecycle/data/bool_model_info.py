import PyBoolNet.FileExchange as FE


def net_inputs(bnet_file, primes_file_out=None):
    primes = FE.bnet2primes(bnet_file, FnamePRIMES=primes_file_out)

    # https://github.com/hklarner/PyBoolNet/blob/master/PyBoolNet/QuineMcCluskey.py
    expressions = {}
    _inputs = []
    for name in primes:
        # name is const
        if primes[name][1] == [{}]:
            expressions[name] = "1"
            continue
        if primes[name][0] == [{}]:
            expressions[name] = "0"
            continue
        _inputs.extend(sorted(set([x for p in primes[name][1] for x in p])))
    return sorted(list(set(_inputs)), key=lambda v: v.upper())


def detect_duplicates(names: list):
    d = dict()
    for n in names:
        _n = n.lower()
        if _n not in d:
            d[_n] = list()
        d[_n].append(n)
    print(d)
    _duplicates = {_k: _v for _k, _v in d.items() if len(_v) > 1}

    return list(_duplicates.values())


if __name__ == "__main__":
    _in = net_inputs("regan2020.bnet", primes_file_out="regan2020_primes.dat")
    duplicates = detect_duplicates(_in)
    if not duplicates:
        print("No duplicates detected.")
        with open("differentiation_model_nodes.dat", "w") as fout:
            [print(_i, file=fout) for _i in _in]
    else:
        print("Go fix your model first.")
        print(f"Duplicates ({len(duplicates)}): {duplicates}")
