import numpy as np

__all__ = ['eval_square', 'eval_semi_cube', 'eval_cube']

"""
    Last modified: 2018-09-28
"""


def eval_square(square):
    """
    Fitness function of the magic square: this code takes into
    account the diagonal sums on each square slice

    :param square: array, the solution vector that represents a magic cube.
    :return:  double, the error value of the input solution vector.
              The mean squared error (MSE) of all each row, column, diagonal
              and space diagonal sum to the magic constant is computed

    Author: Koen van der Blom, Hao Wang, Sander van Rijn
    """
    square, magic_constant = _verify_square(square)

    errors = _calc_square_errors(square, magic_constant)

    mse = np.mean(errors)
    return mse


def eval_semi_cube(cube):
    """
    Fitness function of the perfect magic cube: this code takes into
    account the diagonal sums on each square slice

    :param cube: array, the solution vector that represents a magic cube.
    :return:  double, the error value of the input solution vector.
              The mean squared error (MSE) of all each row, column, diagonal
              and space diagonal sum to the magic constant is computed

    Author: Koen van der Blom, Hao Wang, Sander van Rijn
    """
    cube, magic_constant = _verify_cube(cube)

    errors = np.concatenate([
        _calc_cube_errors(cube, magic_constant, diag=False),
        _calc_space_square_diag_errors(cube, magic_constant)
    ])

    mse = np.mean(errors)
    return mse


def eval_cube(cube):
    """
    Fitness function of the perfect magic cube: this code takes into
    account the diagonal sums on each square slice

    :param cube: array, the solution vector that represents a magic cube.
    :return:  double, the error value of the input solution vector.
              The mean squared error (MSE) of all each row, column, diagonal
              and space diagonal sum to the magic constant is computed

    Author: Koen van der Blom, Hao Wang, Sander van Rijn
    """
    cube, magic_constant = _verify_cube(cube)

    errors = np.concatenate([
        _calc_cube_errors(cube, magic_constant, diag=True),
        _calc_space_square_diag_errors(cube, magic_constant)
    ])

    mse = np.mean(errors)
    return mse


# ------------- Utility Functions ------------

def raise_representation_error(num_numbers, required_numbers, representation):
    missing = required_numbers - set(representation)
    not_belong = set(representation) - required_numbers
    not_belong = None if len(not_belong) == 0 else not_belong
    raise ValueError('Invalid representation! The solution should be a permutation of 1,...,{n}\n'
                     '  Missing numbers: {missing}\n'
                     '  Numbers that do not belong: {not_belong}'.format(n=num_numbers,
                                                                         missing=missing,
                                                                         not_belong=not_belong))


def _verify_square(square):
    n = len(square) ** (1 / 2)
    if np.round(n) ** 2 != len(square):
        raise ValueError('Invalid length! The solution length should be a square number')
    else:
        n = int(np.round(n))

    required_numbers = set(range(1, n**2+1))
    if len(set(square) ^ required_numbers) != 0:
        raise_representation_error(n**2, required_numbers, square)

    magic_constant = n * (n ** 2 + 1) / 2
    square = np.array(square).reshape((n, n))
    return square, magic_constant


def _verify_cube(cube):
    n = len(cube) ** (1 / 3)
    if np.round(n) ** 3 != len(cube):
        raise ValueError("Invalid length! The solution length should be a cubic number")
    else:
        n = int(np.round(n))

    required_numbers = set(range(1, n**3+1))
    if len(set(cube) ^ required_numbers) != 0:
        raise_representation_error(n**3, required_numbers, cube)

    magic_constant = n * (n**3 + 1) / 2
    cube = np.array(cube).reshape((n, n, n))
    return cube, magic_constant


def _calc_square_errors(square, magic_constant, diag=True):

    sums = [
        np.sum(square, axis=0),               # columns
        np.sum(square, axis=1),               # rows
    ]
    if diag:
        sums.extend([
            [np.sum(np.diag(square))],            # diagonal 1
            [np.sum(np.diag(np.rot90(square)))],  # diagonal 2
        ])

    return (np.concatenate(sums)-magic_constant)**2


def _calc_cube_errors(cube, magic_constant, diag=True):

    errors = []
    for square in cube:
        errors = np.concatenate((errors,_calc_square_errors(square, magic_constant, diag=diag)))

    for i in range(cube.shape[1]):
        square = cube[:, i, :]
        sums = [
            np.sum(square, axis=1),               # pillars
        ]
        if diag:
            sums.extend([
                [np.sum(np.diag(square))],            # diagonal 1
                [np.sum(np.diag(np.rot90(square)))],  # diagonal 2
            ])
        sums = np.concatenate(sums)
        errors = np.concatenate((errors, (sums-magic_constant)**2))

    if diag:
        for i in range(cube.shape[2]):
            square = cube[:, :, i]
            sums = np.array([
                np.sum(np.diag(square)),            # diagonal 1
                np.sum(np.diag(np.rot90(square))),  # diagonal 2
            ])
            errors = np.concatenate((errors, (sums-magic_constant)**2))

    return errors


def _calc_space_square_diag_errors(cube, magic_constant):
    n = cube.shape[0]
    space_square1 = np.zeros((n, n))
    space_square2 = np.zeros((n, n))
    for i in range(n):
        space_square1[i, :] = np.diag(cube[:, :, i])
        space_square2[i, :] = np.diag(np.rot90(cube[:, :, i]))

    space_diag_sum = np.array([
        np.sum(np.diag(space_square1)),
        np.sum(np.diag(np.rot90(space_square1))),
        np.sum(np.diag(space_square2)),
        np.sum(np.diag(np.rot90(space_square2)))
    ]).flatten()
    errors = (space_diag_sum - magic_constant) ** 2
    return errors


if __name__ == '__main__':
    perfect_square = [
        2, 7, 6,
        9, 5, 1,
        4, 3, 8,
    ]

    semi_perfect_cube = [
         4, 12, 26,
        11, 25,  6,
        27,  5, 10,

        20,  7, 15,
         9, 14, 19,
        13, 21,  8,

        18, 23,  1,
        22,  3, 17,
         2, 16, 24,
    ]

    perfect_cube = [
         25,  16,  80, 104,  90,
        115,  98,   4,   1,  97,
         42, 111,  85,   2,  75,
         66,  72,  27, 102,  48,
         67,  18, 119, 106,   5,

         91,  77,  71,   6,  70,
         52,  64, 117,  69,  13,
         30, 118,  21, 123,  23,
         26,  39,  92,  44, 114,
        116,  17,  14,  73,  95,

         47,  61,  45,  76,  86,
        107,  43,  38,  33,  94,
         89,  68,  63,  58,  37,
         32,  93,  88,  83,  19,
         40,  50,  81,  65,  79,

         31,  53, 112, 109,  10,
         12,  82,  34,  87, 100,
        103,   3, 105,   8,  96,
        113,  57,   9,  62,  74,
         56, 120,  55,  49,  35,

        121, 108,   7,  20,  59,
         29,  28, 122, 125,  11,
         51,  15,  41, 124,  84,
         78,  54,  99,  24,  60,
         36, 110,  46,  22, 101,
    ]

    all_passed = True

    f = eval_square(perfect_square)
    expected = 0.0
    try:
        assert f == expected
    except AssertionError:
        print('Square:', f, '!=', expected)
        all_passed = False

    f = eval_semi_cube(semi_perfect_cube)
    expected = 0.0
    try:
        assert f == expected
    except AssertionError:
        print('Semi-Cube:', f, '!=', expected)
        all_passed = False

    f = eval_cube(perfect_cube)
    expected = 0.0
    try:
        assert f == expected
    except AssertionError:
        print('Perfect Cube:', f, '!=', expected)
        all_passed = False

    if all_passed:
        print('Test passed')
    else:
        print('Test failed')
