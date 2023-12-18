#!/usr/bin/env python

EPSILON = "epsilon"

SEGMENT = 255
START_SEGMENT = 128

C_NUM = 12

# n -> Nonsingular code
# s -> Singular code
C = {
    1: {
        "n": 0,
        "s": EPSILON
    },
    2: {
        "n": 0,
        "s": 1
    },
    3: {
        "n": 1,
        "s": 0
    },
    4: {
        "n": 1,
        "s": EPSILON
    },
    5: {
        "n": 1,
        "s": 2
    },
    6: {
        "n": 2,
        "s": 1
    },
    7: {
        "n": 2,
        "s": EPSILON
    },
    8: {
        "n": 2,
        "s": 3
    },
    9: {
        "n": 3,
        "s": 2
    },
    10: {
        "n": 3,
        "s": EPSILON
    },
    11: {
        "n": 3,
        "s": 4
    },
    12: {
        "n": 4,
        "s": 3
    }
}

C_NAMES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6',
           'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
