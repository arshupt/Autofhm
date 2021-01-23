import json
import yaml
from yaml import YAMLError, scanner
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import xml.etree.ElementTree as XMLLoader

def etree_to_dict(t):
    d = {t.tag : map(etree_to_dict, t.iterchildren())}
    d.update(('@' + k, v) for k, v in t.attrib.iteritems())
    d['text'] = t.text
    return d

def parse_xml(path) :
    e = None
    try:
        base_tree = XMLLoader.parse(path)
    except FileNotFoundError:
        err = FileNotFoundError
    except PermissionError:
        err = PermissionError
    except Exception as e:
        err = "Random error" + e.args
    return etree_to_dict(base_tree.getroot()), err


def parse_json(path) :
    entitiySetDetails = None
    err = None
    try :
        with open(path) as stream :
            entitiySetDetails = json.load(stream)
    except FileNotFoundError:
        err = FileNotFoundError
    except PermissionError:
        err = PermissionError
    except json.decoder.JSONDecodeError as JSONDecodeError:
        err = JSONDecodeError
    except:
        err = "Unable to parse the file"

    return entitiySetDetails, err


def parse_yaml(path) :
    entitiySetDetails = None
    err = None
    try :
        with open(path) as stream :
            entitiySetDetails = yaml.load(stream, Loader=Loader)
    except PermissionError:
        err = PermissionError
    except FileNotFoundError:
        err = FileNotFoundError
    except YAMLError: 
        err = YAMLError
    except :
        err = "Unable to parse the file"
    return entitiySetDetails, err
