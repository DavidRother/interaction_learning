import imp
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name) + ".py"
    return imp.load_source('', pathname)
