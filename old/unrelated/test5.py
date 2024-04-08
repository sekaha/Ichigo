# https://oeis.org/A122943
# a(n) = A000265(A101695(n)).
# https://oeis.org/A000265 "Remove all factors of 2 from n"
# https://oeis.org/A101695
# https://oeis.org/A014614

k10primes = [
    1048576,
    1572864,
    2359296,
    2621440,
    3538944,
    3670016,
    3932160,
    5308416,
    5505024,
    5767168,
    5898240,
    6553600,
    6815744,
    7962624,
    8257536,
    8650752,
    8847360,
    8912896,
    9175040,
    9830400,
]


def get_p(arr):
    ret = 1

    for v in arr:
        ret *= v

    return ret


def prime_factors(n):
    factors = []
    divisor = 2
    while n > 1:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    return factors


for kprime in k10primes:
    factors = prime_factors(kprime)
    # f"{factors.count(2)}x2",
    # print(" ".join([f"{v}" for v in factors if v != 2]) + ", ")
    print(get_p([v for v in factors if v != 2]))
