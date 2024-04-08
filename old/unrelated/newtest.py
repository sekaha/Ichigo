def prime_factors(n):
    factors = []
    divisor = 2
    while n > 1:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    return factors


def all_prime_factorizations(n, current=[]):
    if n == 1:
        print(current, len(current))
        return
    factors = prime_factors(n)
    for factor in set(factors):
        all_prime_factorizations(n // factor, current + [factor])


# Example usage:
number = int(input("Enter a number to find all prime factorizations: "))
print("All prime factorizations of", number, "are:")
all_prime_factorizations(number)
