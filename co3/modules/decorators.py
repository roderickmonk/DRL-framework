import itertools as it


def call_counter(func):
    def helper(*args, **kwargs):
        helper.count = next(helper._count)
        return func(*args, **kwargs)

    helper._count = it.count(1)
    helper.count = next(helper._count)
    helper.__name__ = func.__name__

    return helper

