def get_p(arr):
    ret = 1

    for v in arr:
        ret *= v

    return ret


def get_k_almost_primes(k, n):
    factors = [2] * k


print(get_p([5] + [2] * 19))

# print(get_k_almost_primes(*map(int, input().split(" "))))
