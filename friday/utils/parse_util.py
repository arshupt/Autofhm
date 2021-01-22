import json
import yaml
from yaml import YAMLError, scanner
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def parse_xml(path) :
    pass

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