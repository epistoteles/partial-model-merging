import numpy as np
import scipy


def extend_cost_matrix(cost_matrix, buffer_size, buffer_min=0.0, buffer_max=1.0):
    """
    Takes in a quadratic cost matrix of shape (a,a):

    xxx
    xxx
    xxx

    and returns the matrix extended with the buffer_max (1) and buffer_min (0) values:

    xxx11
    xxx11
    xxx11
    11100
    11100

    :param cost_matrix: a 2-dimensional np.array()
    :param buffer_min: the minimum value for the buffer (default: 0)
    :param buffer_max: the maximum value for the buffer (default: 1)
    :param buffer_size: the size of the buffer
    :return: the extended cost_matrix of size (a + buffer_size, a + buffer_size)
    """
    size = cost_matrix.shape[0]
    extended_cost_matrix = (
        np.ones((size + buffer_size, size + buffer_size)) * buffer_max
    )
    extended_cost_matrix[0:size, 0:size] = cost_matrix
    extended_cost_matrix[size:, size:] = buffer_min
    return extended_cost_matrix


def print_cost_matrix(cost_matrix, lap_solution=None, buffer_size=None):
    """
    Prints a cost matrix and highlights the values specified in the LAP solution (if given)
    :param cost_matrix: a 2d cost matrix as np.array()
    :param lap_solution: a LAP solution in the form (row_indices, col_indices)
    :param buffer_size: the size of the used buffer
    :return: prints the matrix on the console
    """
    l = cost_matrix.tolist()
    l = [[f" {x:.2f} " for x in line] for line in l]
    if lap_solution is not None:
        for (y, x) in zip(*lap_solution):
            if (
                buffer_size is not None
                and max(x, y) >= cost_matrix.shape[0] - buffer_size
            ):
                if min(x, y) >= cost_matrix.shape[0] - buffer_size:
                    l[y][x] = red(l[y][x])
                else:
                    l[y][x] = yellow(l[y][x])
            else:
                l[y][x] = green(l[y][x])
    # if buffer_size is not None:
    #     for y in range(cost_matrix.shape[0] - buffer_size):
    #         for x in range(cost_matrix.shape[0] - buffer_size):
    #             l[y][x] = purple(l[y][x])
    # for y in range(cost_matrix.shape[0] - buffer_size, cost_matrix.shape[0]):
    #     for x in range(cost_matrix.shape[0]):
    #         l[y][x] = purple(l[y][x])
    # for y in range(cost_matrix.shape[0] - buffer_size):
    #     for x in range(cost_matrix.shape[0] - buffer_size, cost_matrix.shape[0]):
    #         l[y][x] = purple(l[y][x])
    print(f"╭{'─' * (cost_matrix.shape[0] * 6)}╮")
    for line in l:
        print("".join(["│"] + line + ["│"]))
    print(f"╰{'─' * (cost_matrix.shape[0] * 6)}╯")


def green(s):
    """makes a string print with green background"""
    return f"\033[42m{s}\033[0m"


def yellow(s):
    """makes a string print with yellow background"""
    return f"\033[43m{s}\033[0m"


def red(s):
    """makes a string print with red background"""
    return f"\033[41m{s}\033[0m"


def white(s):
    """makes a string print with white background"""
    return f"\033[7m{s}\033[0m"


def grey(s):
    """makes a string print with grey background"""
    return f"\033[100m{s}\033[0m"


def purple(s):
    """makes a string print with purple background"""
    return f"\033[105m{s}\033[0m"
