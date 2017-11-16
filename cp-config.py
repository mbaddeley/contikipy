#!/usr/bin/env python2.7
import itertools as it
from itertools import product, combinations


def TUPLES(config, key):
    """ Turns a dict into a k,v tuple """
    return ((key, t) for t in config[key])


def P_LIST(keys, *args):
    """ Returns product of arg lists as list of dicts """
    return [dict(zip(keys, p)) for p in product(*args)]


def P_FLATTEN(p1, p2):
    flat_dict = p1.copy()   # start with x's keys and values
    flat_dict.update(p2)    # modifies z with y's keys and values. Reurns None.
    return flat_dict


def FLATTEN_ZIP(z):
    flat_l = []
    for t in z:
        flat_d = {}
        for d in t:
            flat_d.update(d)
        flat_l.append(flat_d.copy())
    return flat_l


def P_DICTLIST(d1, d2):
    """ Returns product of two dict lists as a list of dicts """
    dictlist = []
    for p1, p2 in product(d1, d2):
        dictlist += [P_FLATTEN(p1, p2)]
    return dictlist


def DICTLIST_TO_ARGLIST(D):
    """ Turns a list of dicts into a list of makefile args """
    argstring_list = []
    for d in D:
        argstring_list += [" ".join(["=".join([key, str(val).replace(' ', '')])
                                     for key, val in d.items()])]
    return argstring_list


class Config:
    RPL_MODE = ['NS']  # 'STORING' 'NS'
    SDN = [0, 1]
    REROUTE = [1]

    def args(self):
        my_args = P_LIST(['RPL_MODE', 'SDN', 'REROUTE'],
                         self.RPL_MODE, self.SDN, self.REROUTE)
        return my_args


class RplConfig(Config):

    def args(self):
        return 'base config'


class SdnConfig(Config):
    NSU_PERIOD = [5, 30]

    def args(self, mapp):
        return Config.product(self) + ' sdn'


class SingleAppConfig(Config):
    TXNODE = [0]
    NUM_APPS = [1]
    CBR = [1, 5, 20]
    VBR_MIN = [[1], [30]]
    VBR_MAX = [[5], [60]]

    def args(self):
        super_dictlist = Config.args(self)
        cbr_dictlist = P_LIST(['TXNODE', 'NUM_APPS', 'CBR'],
                              self.TXNODE, self.NUM_APPS, self.CBR)
        vbr_dictlist = P_LIST(['TXNODE', 'NUM_APPS', 'VBR_MIN', 'VBR_MAX'],
                              self.TXNODE, self.NUM_APPS, self.VBR_MIN,
                              self.VBR_MAX)
        # EITHER cbr or vbr, not both
        arg_dictlist_cbr = P_DICTLIST(super_dictlist, cbr_dictlist)
        arg_dictlist_vbr = P_DICTLIST(super_dictlist, vbr_dictlist)
        arg_stringlist = DICTLIST_TO_ARGLIST(arg_dictlist_cbr +
                                             arg_dictlist_vbr)
        return arg_stringlist


class MultiAppConfig(Config):
    TXNODE = [3]
    NUM_APPS = [4]
    VBR_MIN = [[1, 1, 1, 20]]
    VBR_MAX = [[5, 10, 20, 20]]

    def args(self):
        super_dictlist = Config.args(self)
        my_dictlist = P_LIST(['TXNODE', 'NUM_APPS'],
                             self.TXNODE, self.NUM_APPS)
        max_dictlist = P_LIST(['VBR_MAX'], self.VBR_MAX)
        min_dictlist = P_LIST(['VBR_MIN'], self.VBR_MIN)
        vbr_dictlist = FLATTEN_ZIP(zip(max_dictlist, min_dictlist))
        arg_dictlist = P_DICTLIST(super_dictlist, my_dictlist)
        arg_dictlist = P_DICTLIST(arg_dictlist, vbr_dictlist)
        arg_stringlist = DICTLIST_TO_ARGLIST(arg_dictlist)
        return arg_stringlist
