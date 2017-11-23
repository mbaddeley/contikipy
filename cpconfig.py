#!/usr/bin/env python2.7
import itertools as it
from itertools import product, combinations
import yaml

# import yaml config
cfg = yaml.load(open("config.yaml", 'r'))


def TUPLES(config, key):
    """ Turns a dict into a k,v tuple """
    return ((key, t) for t in config[key])


def P_LIST(keys, *args):
    """ Returns product of arg lists as list of dicts """
    for p in product(*args):
        print args
    return [dict(zip(keys, p)) for p in product(*args)]


def P_FLATTEN(p1, p2):
    """ Returns a flattened product of two dicts """
    flat_dict = p1.copy()   # start with x's keys and values
    flat_dict.update(p2)    # modifies z with y's keys and values. Reurns None.
    return flat_dict


def FLATTEN_ZIP(z):
    """ Returns a flattened zip() """
    flat_l = []
    for t in z:
        flat_d = {}
        for d in t:
            flat_d.update(d)
        flat_l.append(flat_d.copy())
    return flat_l


def P_DICTLIST(d1, d2):
    """ Returns product of two dictlists lists as a list of dicts """
    dictlist = []
    for p1, p2 in product(d1, d2):
        dictlist += [P_FLATTEN(p1, p2)]
    return dictlist


def DICT_TO_STRING(d):
    return " ".join(["=".join([key, str(val).replace(' ', '')]) for
                    key, val in d.items()])


def DICTLIST_TO_STRINGLIST(D):
    """ Turns a list of dicts into a list of strings """
    argstring_list = []
    for d in D:
        argstring_list += [" ".join(["=".join([key, str(val).replace(' ', '')])
                                     for key, val in d.items()])]
    return argstring_list


class Config:
    SIMLIST = []
    for sim in cfg['simulations']:
        sim['makeargs'] = DICT_TO_STRING(sim['makeargs'])
        SIMLIST.append(sim)

    def simconfig(self):
        print self.SIMLIST
        return self.SIMLIST