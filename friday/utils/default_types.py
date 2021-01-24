from featuretools.variable_types import *

DEFAULT_VARIABLE_TYPES = {
    "categorical": Categorical,
    "numerical": Numeric,
    "boolean": Boolean,
    "discrete": Discrete,
    "index": Index,
    "id": Id,
    "datetime": Datetime,
    "timedelta": Timedelta,
    "text": Text
}