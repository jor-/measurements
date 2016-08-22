import numpy as np

DEFAULT_OPTIONS = {'start':50, 'increment_start':20, 'increment_value':10, 'increment_value_max':50, 'increment_step':1}

def diff(i, start=50, increment_start=20, increment_value=10, increment_value_max=50, increment_step=1):
    options = {'start':start, 'increment_start':increment_start, 'increment_value':increment_value, 'increment_value_max':increment_value_max, 'increment_step':increment_step}

    ## skip diff with no increase
    modulo = i % increment_step
    if modulo != 0:
        i = i - modulo

    ## calculate diff
    if i <= 0:
        return start
    else:
        previous_diff = diff(i - increment_step, **options)
        increment = min(increment_start  + (i / increment_step - 1) * increment_value, increment_value_max)
        return previous_diff + increment


def value(i, **options):
    if i <= 0:
        return 0
    else:
        return value(i-1, **options) + diff(i-1, **options)


def values_fix_length(length, **options):
    return list(map(lambda i: value(i, **options), range(length)))


def values_until_max(max_value, **options):
    values = []
    i = 0
    max_reached = False
    while not max_reached:
        value_i = value(i, **options)
        values.append(value_i)
        max_reached = max_value <= value_i
        i += 1

    return values


def values_TMM(max_value=9300, increment_step=1):
    start = 50 / increment_step
    increment_start = 20 / increment_step
    increment_value = 10 / increment_step
    increment_value_max = 50 / increment_step

    return values_until_max(max_value, start=start, increment_start=increment_start, increment_value=increment_value, increment_value_max=increment_value_max, increment_step=increment_step)

