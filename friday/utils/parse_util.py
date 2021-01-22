import json

def parse_xml(path) :
    pass

def parse_json(path) :
    try:
        with open(path) as f:
            data = json.load(f)
            return data
    except FileNotFoundError as err:
        return err
    except PermissionError as err:
        return err
    except json.decoder.JSONDecodeError:
        return "JSONDecodeError"
    except:
        return "error"

def parse_yaml(path) :
    pass