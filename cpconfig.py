#!/usr/bin/env python2.7
"""Module allows parsing of the yaml config."""
from itertools import product

from pprint import pprint


def TUPLES(config, key):
    """Turn a dict into a k,v tuple."""
    return ((key, t) for t in config[key])


def P_LIST(keys, *args):
    """Return product of arg lists as list of dicts."""
    # for p in product(*args):
    #     print(args)
    return [dict(zip(keys, p)) for p in product(*args)]


def P_FLATTEN(p1, p2):
    """Return a flattened product of two dicts."""
    flat_dict = p1.copy()   # start with x's keys and values
    flat_dict.update(p2)    # modifies z with y's keys and values. Reurns None.
    return flat_dict


def FLATTEN_ZIP(z):
    """Return a flattened zip()."""
    flat_l = []
    for t in z:
        flat_d = {}
        for d in t:
            flat_d.update(d)
        flat_l.append(flat_d.copy())
    return flat_l


def P_DICTLIST(d1, d2):
    """Return product of two dictlists lists as a list of dicts."""
    dictlist = []
    for p1, p2 in product(d1, d2):
        dictlist += [P_FLATTEN(p1, p2)]
    return dictlist


def DICT_TO_STRING(d):
    """Turn a dict into a string."""
    return " ".join(["=".join([key, str(val).replace(' ', '')]) for
                     key, val in d.items()])


def DICTLIST_TO_STRINGLIST(D):
    """Turn a list of dicts into a list of strings."""
    argstring_list = []
    for d in D:
        argstring_list += [" ".join(["=".join([key, str(val).replace(' ', '')])
                                     for key, val in d.items()])]
    return argstring_list


class Config:
    """Config class."""

    cfg = None
    MFLOWS = {}
    SIMS = []

    def __init__(self, cfg):
        """Get YAML config and find simulations."""
        self.cfg = cfg

        # format multiflow makeargs
        if 'multiflow' in self.cfg:
            for config in self.cfg['multiflow']:
                index = config['id']
                # format the makeargs as strings
                self.MFLOWS[index] = ''
                # number of flow applications
                self.MFLOWS[index] = (DICT_TO_STRING(
                                      {'NUM_APPS': config['NUM_APPS']}))
                # the rest of the makeargs for this flow
                self.MFLOWS[index] = " ".join([self.MFLOWS[index],
                                               DICT_TO_STRING(
                                                   config['flows'])])
        # format simulation makeargs
        if 'simulations' in cfg:
            for sim in cfg['simulations']:
                # format the makeargs as strings
                if 'makeargs' in sim and sim['makeargs'] is not None:
                    sim['makeargs'] = DICT_TO_STRING(sim['makeargs'])
                if 'multiflow' in sim and 'multiflow' in cfg:
                    # get multiflow args and add them to makeargs
                    sim['makeargs'] = " ".join([sim['makeargs'],
                                                self.MFLOWS[sim['multiflow']]])
                # add to the sims
                self.SIMS.append(sim)

    def simconfig(self):
        """Sim configuration."""
        if 'simulations' in self.cfg:
            return self.SIMS

    def compareconfig(self):
        """Analysis configuration."""
        # print(self.cfg)
        if 'compare' in self.cfg:
            if self.cfg['compare'] is not None:
                return self.cfg['compare'][0]
