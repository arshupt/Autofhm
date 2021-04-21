import pandas as pd
import numpy as np

def index_util(df) :
    df = df.reindex()
    df = df.reset_index(inplace=True)
    return df

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

class Interruptable_cross_val_score(threading.Thread):

    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self)
        self.args = args
        self.kwargs = kwargs
        self.result = -float('inf')
        self._stopevent = threading.Event()
        self.daemon = True 

    def stop(self):

        self._stopevent.set()
        threading.Thread.join(self)

    def run(self):

        threading.current_thread().name = 'MainThread'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.result = cross_val_score(*self.args, **self.kwargs)
        except Exception as e:
            pass

def _wrapped_cross_val_score(sklearn_pipeline, features, classes,
                             cv, scoring_function, sample_weight, max_eval_time_mins):

    max_time_seconds = max(int(max_eval_time_mins * 60), 1)
    sample_weight_dict = set_sample_weight(sklearn_pipeline.steps, sample_weight)

    tmp_it = Interruptable_cross_val_score(clone(sklearn_pipeline), features, classes,
        scoring=scoring_function,  cv=cv, n_jobs=1, verbose=0, fit_params=sample_weight_dict)

    tmp_it.start()
    tmp_it.join(max_time_seconds)

    if tmp_it.isAlive():
        resulting_score = 'Timeout'
    else:
        resulting_score = np.mean(tmp_it.result)

    tmp_it.stop()

    return resulting_score