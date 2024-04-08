chars = "qwertyuiopasdfghjkl;zxcvbnm,./"
count = 0

# test all possible swaps
for i, k1 in enumerate(chars):
    for k2 in chars[i + 1 :]:
        print(k1 + k2)
        count += 1

print(count)
