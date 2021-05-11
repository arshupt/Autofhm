import inspect
import warnings
import threading
from collections import defaultdict
from stopit import threading_timeoutable, TimeoutException

import deap
from deap import tools, gp
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin

from .metrics import metrics

def pareto_eq(ind1, ind2):
            return np.allclose(ind1.fitness.values, ind2.fitness.values)

def create_dataframe_from_entity(entity) :
    
    dataframe_path = entity["dataframe_path"]
    dataframe = None
    try :
        dataframe = pd.read_csv(dataframe_path)
    except Exception as e :
        print("Exception {} happended while reading the CSV file".format(e))

    return dataframe, entity['variable_types']

class Operator(object):

    root = False  
    import_hash = None
    sklearn_class = None
    arg_types = None


class ARGType(object):

    pass

class Output_Array(object):
    
    pass

def source_decode(sourcecode):

    tmp_path = sourcecode.split('.')
    op_str = tmp_path.pop()
    import_str = '.'.join(tmp_path)
    try:
        exec('from {} import {}'.format(import_str, op_str))
        op_obj = eval(op_str)
    except ImportError:
        op_obj = None

    return import_str, op_str, op_obj


def findArgTypes(classname, prange, BaseClass=ARGType):
    return type(classname, (BaseClass,), {'values': prange})


def findOperatorClass(opsourse, opdict, BaseClass=Operator, ArgBaseClass=ARGType):
    class_profile = {}
    dep_op_list = {}
    dep_op_type = {}
    import_str, op_str, op_obj = source_decode(opsourse)

    if not op_obj:
        return None, None
    else:
        
        class_profile['root'] = True
        optype = "Classifier or Regressor"
        @classmethod
        def op_type(cls):
            return optype

        class_profile['type'] = op_type
        class_profile['sklearn_class'] = op_obj
        import_hash = {}
        import_hash[import_str] = [op_str]
        arg_types = []

        for pname in sorted(opdict.keys()):
            prange = opdict[pname]
            classname = '{}__{}'.format(op_str, pname)
            arg_types.append(findArgTypes(classname, prange, ArgBaseClass))
        class_profile['arg_types'] = tuple(arg_types)
        class_profile['import_hash'] = import_hash
        class_profile['dep_op_list'] = dep_op_list
        class_profile['dep_op_type'] = dep_op_type

        @classmethod
        def parameter_types(cls):
            return ([np.ndarray] + arg_types, np.ndarray)

        class_profile['parameter_types'] = parameter_types

        @classmethod
        def export(cls, *args):
            op_arguments = []

            if dep_op_list:
                dep_op_arguments = {}

            for arg_class, arg_value in zip(arg_types, args):
                aname_split = arg_class.__name__.split('__')
                if isinstance(arg_value, str):
                    arg_value = '\"{}\"'.format(arg_value)
                if len(aname_split) == 2: 
                    op_arguments.append("{}={}".format(aname_split[-1], arg_value))
                else:
                    if aname_split[1] not in dep_op_arguments:
                        dep_op_arguments[aname_split[1]] = []
                    dep_op_arguments[aname_split[1]].append("{}={}".format(aname_split[-1], arg_value))

            tmp_op_args = []
            if dep_op_list:
                for dep_op_pname, dep_op_str in dep_op_list.items():
                    arg_value = dep_op_str 
                    doptype = dep_op_type[dep_op_pname]
                    if inspect.isclass(doptype):
                        if issubclass(doptype, BaseEstimator) or \
                            issubclass(doptype, ClassifierMixin) or \
                            issubclass(doptype, RegressorMixin) :
                            arg_value = "{}({})".format(dep_op_str, ", ".join(dep_op_arguments[dep_op_str]))
                    tmp_op_args.append("{}={}".format(dep_op_pname, arg_value))
            op_arguments = tmp_op_args + op_arguments
            return "{}({})".format(op_obj.__name__, ", ".join(op_arguments))

        class_profile['export'] = export

        op_classname = 'SKLEARN_{}'.format(op_str)
        op_class = type(op_classname, (BaseClass,), class_profile)
        op_class.__name__ = op_str
        return op_class, arg_types


def get_by_name(opname, operators):
    
    ret_op_classes = [op for op in operators if op.__name__ == opname]

    if len(ret_op_classes) == 0:
        raise TypeError('Cannot found operator {} in operator dictionary'.format(opname))
    elif len(ret_op_classes) > 1:
        raise ValueError(
            'Found duplicate operators {} in operator dictionary. Please check '
            'your dictionary file.'.format(opname)
        )
    ret_op_class = ret_op_classes[0]
    return ret_op_class


def expr_to_tree(ind, pset):
    
    def prim_to_list(prim, args):
        if isinstance(prim, deap.gp.Terminal):
            if prim.name in pset.context:
                return pset.context[prim.name]
            else:
                return prim.value

        return [prim.name] + args

    tree = []
    stack = []
    for node in ind:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            tree = prim_to_list(prim, args)
            if len(stack) == 0:
                break  
            stack[-1][1].append(tree)
    return tree


def generate_pipeline_code(pipeline_tree, operators):
    steps = _process_operator(pipeline_tree, operators)
    pipeline_text = ''.join(steps)
    return pipeline_text


def _process_operator(operator, operators, depth=0):
    steps = []
    op_name = operator[0]

    input_name, args = operator[1], operator[2:]
    tpot_op = get_by_name(op_name, operators)

    if input_name != 'input_matrix':
        steps.extend(_process_operator(input_name, operators, depth + 1))

    steps.append(tpot_op.export(*args))
    return steps




@threading_timeoutable(default="Timeout")
def cv_score(model, features, targets, cv, scoring_function):
    folds = KFold(n_splits=cv)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            cv_score = []
            features = features.to_numpy()
            targets = targets.to_numpy()

            for train_idx, test_idx in folds.split(features):

                X_train, X_test, y_train, y_test =features[train_idx],features[test_idx], targets[train_idx], targets[test_idx]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scorer = metrics[scoring_function]
                cv_score.append(scorer(y_test, y_pred))

        cv_score = np.array(cv_score)
        print(cv_score)
        nz = np.count_nonzero(np.isnan(cv_score))
        if len(cv_score) - nz == 0 :
            return -float('inf')
        else :
            return np.nanmean(cv_score)
    except TimeoutException:
        return "Timeout"
    except Exception as e:
        print(e)
        exit()
        return -float('inf')


