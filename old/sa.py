import numpy as np
from classifier import classifier, keyboard
from math import ceil, exp, log
from random import shuffle, random

with open("ngrams/bigrams.txt") as f:
    freqs = {a: int(b) for l in f for a, b in [l.strip().split("\t")]}

with open("ngrams/bigrams.txt") as f:
    total_chars = sum(
        [
            int(l.strip().split("\t")[1])
            for l in f
            if all(
                [
                    c in "qwertyuiopasdfghjkl;zxcvbmn,./"
                    for c in l.strip().split("\t")[0]
                ]
            )
        ]
    )


class optimizer:
    def __init__(self):
        self.t0 = 0
        self.cooling_schedule = "default"
        self.keyboard = keyboard()  # ["qwertyuiop", "asdfghjkl'", "zxcvbmn,.-"])
        self.classifier = classifier()
        self.bg_scores = {bg: 0 for bg in self.keyboard.get_ngrams(2)}
        self.new_bg_scores = {}
        self.fitness = 0
        self.prev_fitness = 0
        self.a = 0.99

        self.get_fitness()
        self.accept()
        self.temp = self.get_initial_temperature(0.8, 0.01)
        self.stopping_point = self.get_stopping_point()

    def accept(self):
        self.bg_scores.update(self.new_bg_scores)

    def reject(self):
        self.keyboard.undo_swap()
        self.fitness = self.prev_fitness
        self.new_bg_scores = {}

    def get_initial_temperature(self, x0, epsilon):
        # An initial guess for t1
        tn = 500_000_000_000
        acceptance_probability = 0

        # Repeat guess
        while abs(acceptance_probability - x0) > epsilon:
            energies = []

            # test all possible swaps
            for i, k1 in enumerate(self.keyboard.chars[:-1]):
                for k2 in self.keyboard.chars[i + 1 :]:
                    self.keyboard.swap(k1, k2)
                    self.get_fitness()

                    delta = self.fitness - self.prev_fitness

                    # Keep track of transition energies for each positive transition
                    if delta > 0:
                        energies.append(self.fitness)

                    self.reject()

            # Calculate acceptance probability
            acceptance_probability = sum(
                [exp(-(e_after / tn)) for e_after in energies]
            ) / (len(energies) * exp(-(self.prev_fitness / tn)))

            tn = tn * (log(acceptance_probability) / log(x0))

        return tn

    def cool(self):
        self.temp *= self.a

    # Calculate a stopping time for the annealing process based on the number of swaps (coupon collector's problem).
    def get_stopping_point(self):
        swaps = self.keyboard.key_count * (self.keyboard.key_count - 1) / 2
        euler_mascheroni = 0.5772156649

        return ceil(swaps * (log(swaps) + euler_mascheroni) + 0.5)

    def get_fitness(self):
        self.prev_fitness = self.fitness
        bgs = self.keyboard.get_ngrams(2)

        for bg in bgs:
            if not any(
                [
                    c in "QWERTYUIOPASDGFHJKLZXCVBN}{}[]M!@#$%^&*()12+_3456890"
                    for c in bg
                ]
            ):
                freq = max(277.286496350365, freqs.get(bg, 0))

                features = (
                    freq,
                    self.classifier.is_pinky(bg[0]),
                    self.classifier.is_ring(bg[0]),
                    self.classifier.is_middle(bg[0]),
                    #
                    self.classifier.is_index(bg[0]),
                    self.classifier.is_pinky(bg[1]),
                    self.classifier.is_ring(bg[1]),
                    #
                    self.classifier.is_middle(bg[1]),
                    self.classifier.is_index(bg[1]),
                    self.classifier.is_bottom(bg[0]),
                    #
                    self.classifier.is_homerow(bg[0]),
                    self.classifier.is_top(bg[0]),
                    self.classifier.is_bottom(bg[1]),
                    #
                    self.classifier.is_homerow(bg[1]),
                    self.classifier.is_top(bg[1]),
                    self.classifier.same_finger(bg),
                    #
                    self.classifier.same_hand(bg),
                )

                predicted_time = self.predict_time(features)

                self.new_bg_scores[bg] = predicted_time * freq
                delta = self.new_bg_scores[bg] - self.bg_scores[bg]

                self.fitness += delta

    def optimize(self):
        stays = 0

        while stays < self.stopping_point:
            # markov chain
            for _ in range(30):
                self.keyboard.random_swap()
                self.get_fitness()
                delta = self.fitness - self.prev_fitness

                # Metropolis criterion
                if delta < 0 or (delta > 0 and random() < exp(-delta / self.temp)):
                    self.accept()
                    stays = 0
                else:
                    self.reject()
                    stays += 1

            self.cool()
            print(self.fitness)
            print(self.keyboard)

    def predict_time(self, features):
        p = [
            -0.7682108331636368,
            -264.40536985118473,
            15.355009883013318,
            2.9667635554004526,
            2.694426663363555,
            3.0817479673833095,
            6.305055236083264,
            -0.2550759549938028,
            2.707111640534569,
            3.1790186296929672,
            3.109863985557762,
            3.0618316684471454,
            2.8379160497068234,
            2.967931054215258,
            3.395488171235601,
            3.1217941339400954,
            2.6714647962009326,
            2.463074297313445,
            2.46590623919877,
            2.9983796032250876,
            6.211914296456321,
            -0.9535342052633862,
            2.686727078269682,
            3.071004441859457,
            2.499470596736914,
            2.967793769170462,
            3.089659778801403,
            3.51784381224073,
            2.660419807844165,
            2.613862577071813,
            2.8096477207349193,
            3.147574940539819,
            2.9840007806527735,
            3.4545124904892246,
            6.368110606331885,
            0.15740997698661036,
            3.0686905910412423,
            3.530305415402989,
            3.0147409530354574,
            2.959546132034918,
            2.9443030473004113,
            3.530305416971417,
            3.0147409543660952,
            2.95954612990572,
            2.944303044705876,
            3.486134956680639,
            -5.151529374460863,
            0.1295575767182749,
            -2.792585899352071,
            -3.32459582803057,
            -3.336220733389801,
            -4.64514250945328,
        ]

        freq, is_sfb, same_hand = features[0], features[15], features[16]
        bottom1, home1, top1, bottom2, home2, top2 = features[9:15]
        is_pinky1, is_ring1, is_middle1, is_index1 = features[1:5]
        is_pinky2, is_ring2, is_middle2, is_index2 = features[5:9]

        freq_pen = p[0] * np.log(freq + p[1]) + p[2]

        # SHB weighting
        shb_finger1_row_pen = p[3] * bottom1 + p[4] * home1 + p[5] * top1
        shb_finger2_row_pen = p[6] * bottom2 + p[7] * home2 + p[8] * top2

        shb_finger1_col_pen = (
            p[9] * is_pinky1 + p[10] * is_ring1 + p[11] * is_middle1 + p[12] * is_index1
        )
        shb_finger2_col_pen = (
            p[13] * is_pinky2
            + p[14] * is_ring2
            + p[15] * is_middle2
            + p[16] * is_index2
        )
        shb_row_pen = shb_finger1_row_pen + shb_finger2_row_pen
        shb_col_pen = shb_finger1_col_pen + shb_finger2_col_pen

        shb_finger_pen = shb_row_pen * shb_col_pen

        # alt weighting
        alt_finger1_row_pen = p[17] * bottom1 + p[18] * home1 + p[19] * top1
        alt_finger2_row_pen = p[20] * bottom2 + p[21] * home2 + p[22] * top2

        alt_finger1_col_pen = (
            p[23] * is_pinky1
            + p[24] * is_ring1
            + p[25] * is_middle1
            + p[26] * is_index1
        )
        alt_finger2_col_pen = (
            p[27] * is_pinky2
            + p[28] * is_ring2
            + p[29] * is_middle2
            + p[30] * is_index2
        )
        alt_finger1_row_pen = alt_finger1_row_pen + alt_finger2_row_pen
        alt_finger1_col_pen = alt_finger1_col_pen + alt_finger2_col_pen

        alt_finger_pen = alt_finger1_row_pen * alt_finger1_col_pen

        # Finger weighting
        sfb_finger1_row_pen = p[31] * bottom1 + p[32] * home1 + p[33] * top1
        sfb_finger2_row_pen = p[34] * bottom2 + p[35] * home2 + p[36] * top2

        sfb_finger1_col_pen = (
            p[37] * is_pinky1
            + p[38] * is_ring1
            + p[39] * is_middle1
            + p[40] * is_index1
        )
        sfb_finger2_col_pen = (
            p[41] * is_pinky2
            + p[42] * is_ring2
            + p[43] * is_middle2
            + p[44] * is_index2
        )
        sfb_finger1_row_pen = sfb_finger1_row_pen + sfb_finger2_row_pen
        sfb_finger1_col_pen = sfb_finger1_col_pen + sfb_finger2_col_pen

        sfb_finger_pen = sfb_finger1_row_pen * sfb_finger1_col_pen

        # Base finger pen
        base_pen = (p[45] * bottom2 + p[46] * home2 + p[47] * top2) * (
            p[48] * is_pinky2
            + p[49] * is_ring2
            + p[50] * is_middle2
            + p[51] * is_index2
        )

        # BG Type Classification
        same_hand_weight = same_hand * (1 - is_sfb) * shb_finger_pen
        sfb_weight = is_sfb * sfb_finger_pen
        alt_weight = (1 - same_hand) * alt_finger_pen

        # print(freq)
        # print(same_hand_weight, sfb_weight, alt_weight, base_pen, freq_pen)
        return (same_hand_weight + sfb_weight + alt_weight + base_pen) * freq_pen


best_keeb = None
best_score = float("inf")

for _ in range(10):
    o = optimizer()
    o.optimize()

    print("Fitness", int(o.fitness))
    print("Chars", total_chars)
    print("WPM", (total_chars / 5) / ((o.fitness) / 60 / 1000))

    if o.fitness < best_score:
        best_score = o.fitness
        print("new best")
        best_keeb = o.keyboard

print("best score")
print(best_keeb)
