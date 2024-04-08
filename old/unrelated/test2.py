# K-almost prime
from time import time


def get_kalmost_primes(k, n):
    start = time()
    kalmost_primes = []
    m = 2

    while len(kalmost_primes) < n:
        divisor = 3
        factor_count = 0
        tmp = m

        # remove all temp
        while tmp % divisor == 0:
            tmp //= divisor
            factor_count += 1

        while tmp > 1 and divisor <= int(m**0.5):
            while tmp % divisor == 0:
                tmp //= divisor
                factor_count += 1
            divisor += 2

            if factor_count == k:
                if tmp <= 1:
                    kalmost_primes.append(m)
                    print(f"added! @ {(time() - start)} seconds")
                break

        m += 1

    print((time() - start) / 60)
    return kalmost_primes


print(get_kalmost_primes(*map(int, input().split(" "))))
