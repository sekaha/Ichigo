# Python3 Program to print first
# n numbers that are k-primes
import math
from time import time


# A function to count all prime
# factors of a given number
def countPrimeFactors(n):
    count = 0

    # Count the number of
    # 2s that divide n
    while n % 2 == 0:
        n = n / 2
        count += 1

    # n must be odd at this point.
    # So we can skip one
    # element (Note i = i +2)
    i = 3
    while i <= math.sqrt(n):

        # While i divides n,
        # count i and divide n
        while n % i == 0:
            n = n / i
            count += 1
        i = i + 2

    # This condition is to handle
    # the case when n is a
    # prime number greater than 2
    if n > 2:
        count += 1

    return count


# A function to print the
# first n numbers that are
# k-almost primes.
def printKAlmostPrimes(k, n):
    i = 1
    num = 2
    while i <= n:

        # Print this number if
        # it is k-prime
        if countPrimeFactors(num) == k:
            print(num, end="")
            print(" ", end="")

            # Increment count of
            # k-primes printed
            # so far
            i += 1
        num += 1
    return


# Driver Code
s = time()
n = 5
k = 20
print("First n k-almost prime numbers:")
printKAlmostPrimes(k, n)
print(time() - s)

# This code is contributed by mits
