import param

_author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"


class TupleList(param.Parameter):
    """Parameter whose value is a list of fixed-length tuples"""

    __slots__ = ['class_', 'bounds', 'length']

    def __init__(
            self, default=[], bounds=(0, None), class_=None, length=None,
            instantiate=True, **params):
        super(TupleList, self).__init__(
            default=default, instantiate=True, **params)
        self.class_ = class_
        self.bounds = bounds
        if length is None and len(default):
            self.length = len(default[0])
        elif length is None and not len(default):
            raise ValueError(
                '{}: length must be specified if no default is supplied'
                ''.format(self._attrib_name))
        else:
            self.length = length
        self._check_bounds(default)

    def __set__(self, obj, val):
        self._check_bounds(val)
        super(TupleList, self).__set__(obj, val)

    def _check_bounds(self, val):
        if self.allow_None and val is None:
            return
        if not isinstance(val, list):
            raise ValueError(
                "List '{}' must be a list.".format(self._attrib_name))
        if self.bounds is not None:
            min_length, max_length = self.bounds
            if min_length is not None and max_length is not None:
                if not (min_length <= len(val) <= max_length):
                    raise ValueError(
                        "{}: list length must be between {} and {} (inclusive)"
                        "".format(self._attrib_name, min_length, max_length))
            elif min_length is not None:
                if not min_length <= len(val):
                    raise ValueError(
                        "{}: list length must be at least {}."
                        "".format(self._attrib_name, min_length))
            elif max_length is not None:
                if not len(val) <= max_length:
                    raise ValueError(
                        "{}: list length must be at most {}."
                        "".format(self._attrib_name, max_length))
        self._check_type(val)

    def _check_type(self, val):
        for v in val:
            assert isinstance(v, tuple), "{} is not a tuple".format(val)
            assert len(v) == self.length, (
                '{} does not have length {}'.format(v, self.length))
            if self.class_ is not None:
                for vv in v:
                    assert isinstance(vv, self.class_), (
                        '{} is not a {}'.format(vv, self.class_))
