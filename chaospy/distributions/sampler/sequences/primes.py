"""
Create all primes bellow a certain threshold.

Examples::

    >>> create_primes(1)
    []
    >>> create_primes(2)
    [2]
    >>> create_primes(3)
    [2, 3]
    >>> create_primes(20)
    [2, 3, 5, 7, 11, 13, 17, 19]
"""


def create_primes(threshold):
    """
    Generate prime values using sieve of Eratosthenes method.

    Args:
        threshold (int):
            The upper bound for the size of the prime values.

    Returns (List[int]):
        All primes from 2 and up to ``threshold``.
    """
    if threshold == 2:
        return [2]

    elif threshold < 2:
        return []

    numbers = list(range(3, threshold+1, 2))
    root_of_threshold = threshold ** 0.5
    half = int((threshold+1)/2-1)
    idx = 0
    counter = 3
    while counter <= root_of_threshold:
        if numbers[idx]:
            idy = int((counter*counter-3)/2)
            numbers[idy] = 0
            while idy < half:
                numbers[idy] = 0
                idy += counter
        idx += 1
        counter = 2*idx+3
    return [2] + [number for number in numbers if number]
