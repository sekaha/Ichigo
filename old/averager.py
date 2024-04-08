import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import log
import numpy as np
import classifier

x = []
y = []
c = []

wpm = 47

bigrams = {
    k: int(v) for (k, v) in [l.strip().split("\t") for l in open("ngrams/bigrams.txt")]
}


title = f"Relationship Between English Bigram Frequency and Typing Time at >{wpm} WPM"

if wpm == 0:
    title = "Relationship Between English Bigram Frequency and Typing Time"

plt.title(title)

with open(f"nstrokes/bistrokes.txt") as file:
    for l in file:
        bistroke = l[:2]
        times = list(map(int, l[3:].split(", ")))
        avg = int(sum(times) / len(times))

        # print(f"{bistroke}: {avg}")
        if not any(
            [c in '~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:"ZXCVBNM<>?' for c in bistroke]
        ):
            y.append(avg)
            x.append(bigrams[bistroke])
            c.append(
                "red"
                if classifier.inwards_rotation(bistroke)
                else "black" if classifier.outwards_rotation(bistroke) else "yellow"
            )
            # print("".join(bistroke[::-1]))
            # if (
            #    bistroke in list(bigrams.keys())[:20]
            #    or "".join(bistroke[::-1]) in list(bigrams.keys())[:20]
            # ):
            # plt.annotate(bistroke, xy=(bigrams[bistroke], avg))
"""
skipgrams = {
    k: int(v) for (k, v) in [l.strip().split("\t") for l in open("ngrams/1-skip.txt")]
}

with open("nstrokes/1-skip_100.txt") as file:
    for l in file:
        ngram = l[:2]
        times = list(map(int, l[3:].split(", ")))
        avg = int(sum(times) / len(times))
        # print(f"{bistroke}: {avg}")
        # if not any([c.isupper() for c in ngram]):
        y.append(avg)
        x.append(skipgrams[ngram])
        c.append("red" if ngram[0] == ngram[1] else "#1f77b4")
        plt.annotate(ngram, xy=(skipgrams[ngram], avg))

trigrams = {
    k: int(v) for (k, v) in [l.strip().split("\t") for l in open("ngrams/trigrams.txt")]
}

with open("nstrokes/tristrokes_90.txt") as file:
    for l in file:
        ngram = l[:3]
        times = list(map(int, l[4:].split(", ")))
        avg = int(sum(times) / len(times))
        # print(f"{bistroke}: {avg}")
        if not any([c.isupper() for c in ngram]):
            # there's some typos that somehow don't even present themselves in my corpus omg
            if ngram in trigrams:
                x.append(trigrams[ngram])
            else:
                x.append(0)
            y.append(avg)
            c.append("red" if ngram[0] == ngram[1] == ngram[2] else "#1f77b4")
            # plt.annotate(ngram, xy=(x[-1], y[-1]))
"""


def log_func(x, a, b, c):
    return a * np.log(x + b) + c


popt, pcov = curve_fit(log_func, x, y)
new_y = log_func(np.sort(x), *popt)
print(*popt)
plt.scatter(x, y, c=c, s=20)
plt.plot(np.sort(x), new_y, c="#ff6361")
plt.xlabel("Number of Occurrences in Corpus ")
plt.ylabel("Average Typing Time (Milliseconds)")
plt.xscale("log")
plt.show()

import tikzplotlib

tikzplotlib.save("test.tex")
