import warnings
import numpy as np
import pandas as pd

class ClassNameDescriptor(object) :
    def __get__(self, instance, class_) :
        return camel_to_snake(class_.__name__)

class Variables(object) :
    type_string = ClassNameDescriptor()
    _default_pandas_dtype = object
    def __init__(self, id, entity, name=None, description=None) :
        assert isinstance(id, str), "Variable id must be a string"
        self.id = id
        self._name = name
        self.entity_id = entity.id
        self._description = description
        assert entity.entitySet is not None, "Entity must contain reference to Entity Set"
        self.entity = entity
        if self.id not in self.entity.df :
            default_dtype = self._default_pandas_dtype
            if default_dtype == np.datetime64 :
                default_dtype = 'datetime64[ns'
            if default_dtype == np.timedelta64 :
                default_dtype = 'timedelta64[ns'
        else :
            default_dtype = self.entity.df[self.id].dtype
        self._interesting_values = pd.Series(dtype=default_dtype)

    def __eq__(self, other, deep=False) :
        eqaul = isinstance(other, self.__class__) \
            and self.id == other.id \
            and self.entity_id == other.entity_id
        if deep :
            return eqaul and set(self._interesting_values.values) == set(other._interesting_values.values)
        else :
            return eqaul
    
    def __hash__(self) :
        return hash((self.id, self.entity_id))

    def __repr__(self) :
        return u"<Variable: {} (dtype = {})>".format(self.name, self.type_string)

    @property
    def entitySet(self) :
        return self.entity.entitySet

    @property 
    def name(self) :
        return self._name if self._name is not None else self.id

    @name.setter 
    def name(self, name) :
        self._name = name

    @property 
    def description(self) :
        return self._description if self._description is not None else '{} type'.format(self.name)

    @description.setter 
    def description(self, description) :
        if(self._description != description) :
            self.entity.entitySet.reset_data_description()
        self._description = description

    @property 
    def dtype(self) :
        return self.type_string if self.type_string is not None else "Generic Type"

    @property 
    def interesting_values(self) :
        return self._interesting_values
    
    @interesting_values.setter 
    def interesting_values(self, intersting_vales) :
        self._interesting_values = pd.Series(intersting_vales, dtype=self._interesting_values.dtype)

    @property
    def series(self) :
        return self.entity.df[self.id]

    @classmethod
    def create_from(cls, variable) :
        return cls(id=variable.id, name=variable.name, entity=variable.entity)
    
    def to_description(self) :
        return {
            'id': self.id,
            'type': {
                'value': self.type_string,
            },
            'properties': {
                'name': self.name,
                'description': self.description,
                'entity': self.entity,
                'interesting_values': self._interesting_values.to_json(),
            },
        }

class Numeric(Variables) :
    _default_pandas_dtype = float
    def __init__(self, id, entity, name=None, range=None, start_inclusive=True, end_inclusive=False) :
        super(Numeric, self).__init__(id, entity, name=name)
        self.range = range or []
        self.start_inclusive = start_inclusive
        self.end_inclusive = end_inclusive
    
    def to_description(self) :
        description = super(Numeric, self).to_description()
        description['type'].update({
            'range': self.range,
            'start_inclusive': self.start_inclusive,
            'end_inclusive': self.end_inclusive,
        })
        return description

class Discrete(Variables) :
    def __init__(self, id, entity, name=None) :
        super(Discrete, self).__init__(id, entity, name)
    
    @property 
    def interesting_values(self) :
        return self._interesting_values

    @interesting_values.setter 
    def interesting_values(self, values) :
        seen = set()
        addSeen = seen.add
        self._interesting_values = pd.Series([value for value in values if not (value in seen or addSeen(value))], dtype=self._interesting_values.dtype)

class Categorical(Variables) :
    def __init__(self, id, entity, name=None, categories=None) :
        super(Categorical, self).__init__(id, entity, name=name)
        self.categories = categories or []

    def to_description(self) :
        description = super(Categorical, self).to_description()
        description['type'].update({
            'categories': self.categories
            })
        return description

class Boolean(Variables) :
    def __init__(self, id, entity, name=None, true_values=None, false_values) :
        super(Boolean, self).__init__(id, entity, name=name)
        true_default = [ 1, "true", True, "True", 't', 'T', 'yes', 'Yes']
        false_default = [ 0, "false", False, 'False', 'f', 'F', 'no', 'No']
        self.true_values = true_values or true_default
        self.false_values = false_values or false_default

    def to_description(self) :
        description = super(Boolean, self).to_description()
        description['type'].update({
            'true_values': self.true_values,
            'false_values': self.false_values,
        })
        return description

class Datetime(Variables) :
    def __init__(self, id, entity, name=None, format=None) :
        super(Datetime, self).__init__(id, entity, name=name)
        self.format = format
        self._default_pandas_dtype = np.datetime64
    
    def __repr__(self) :
        return u"<Variable: {} (dtype: {}, format: {})>".format(self.name, self.type_string, self.format)
        
    def to_description(self) :
        description = super(Datetime, self).to_description()
        description['type'].update({
            'format': self.format,
        })
        return description

class Timedelta(Variables) :
    def __init__(self, id, entity, name=None, range=None, start_inclusive=True, end_inclusive=False) :
        super(Timedelta, self).__init__(id, entity, name=name)
        self.range = range or []
        self.start_inclusive = start_inclusive
        self.end_inclusive = end_inclusive

    def to_description(self) :
        description = super(Timedelta, self).to_description()
        description['type'].update({
            'range': self.range,
            'start_inclusive': self.start_inclusive,
            'end_inclusive': self.end_inclusive
        })
        return description


DEFAULT_DTYPE_VALUES = {
    np.datetime64: pd.Timestamp.now(),
    np.timedelta64: pd.Timedelta('1d'),
    int: 1,
    float: 1.1,
    object: 'obj',
    bool: False,
    str: 'string',
}