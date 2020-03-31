import itertools


def batched(iterable, n):
    counter = itertools.count()

    for _, group in itertools.groupby(iterable, lambda _: next(counter) // n):
        yield list(group)


def files(path):
    return (p for p in path.iterdir() if p.is_file())
