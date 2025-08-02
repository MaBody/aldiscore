from abc import ABC

############### Abstract Superclasses ###############


class Feature(ABC):
    def compute(self):
        raise NotImplementedError()


class SingletonFeature(Feature):
    """Returns a scalar."""

    pass


class SequenceFeature(Feature):
    """Returns a k-dimensional array."""

    pass


class CombinatorialFeature(Feature):
    """Returns a max{x , k*(k-1)/2}-dimensional array."""

    pass


############### Implementations ###############


# class NumberOfSequences()
