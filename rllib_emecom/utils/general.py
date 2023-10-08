import datetime
from torchviz import make_dot


def progress_bar_string(complete, total, bar_len=20, time_elapsed=None, with_percent=True):
    if complete == total:
        equals = '=' * bar_len
        percent = '100% ' if with_percent else ''
        time = f' Time taken: {datetime.timedelta(seconds=int(time_elapsed))}' if time_elapsed else ''
        return f'{percent}[{equals}]{time}'
    else:
        n = int((complete / total) * bar_len)
        equals = '=' * (n)
        dots = '.' * (bar_len - n - 1)
        p = int((complete / total) * 100)
        percent = f'{p:3d}% ' if with_percent else ''
        if time_elapsed:
            expected_time = (total / complete) * time_elapsed
            eta = expected_time - time_elapsed
            time = f' ETA: {datetime.timedelta(seconds=int(eta))}'
        else:
            time = ''
        return f'{percent}[{equals}>{dots}]{time}'


def calc_closest_factors(c: int):
    """Calculate the closest two factors of c.

    Returns:
      [int, int]: The two factors of c that are closest; in other words, the
        closest two integers for which a*b=c. If c is a perfect square, the
        result will be [sqrt(c), sqrt(c)]; if c is a prime number, the result
        will be [1, c]. The first number will always be the smallest, if they
        are not equal.
    """
    if c // 1 != c:
        raise TypeError("c must be an integer.")

    a, b, i = 1, c, 0
    while a < b:
        i += 1
        if c % i == 0:
            a = i
            b = c // a

    return [b, a]


def get_best_grid(c: int, thresh=4, max_empty=4) -> tuple:
    """
    Calculate the best grid for a given number of elements.
    """
    rows, cols = calc_closest_factors(c)
    if cols / rows >= thresh:
        return max([
            calc_closest_factors(c + dc)
            for dc in range(c, c + max_empty)
        ], key=lambda x: x[1] / x[0])

    return rows, cols


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def render_computational_graph(tensor, named_params, output_name='output', **kwargs):
    graph = make_dot(tensor, params=named_params, **kwargs)
    graph.render(output_name, format='pdf', cleanup=True)
