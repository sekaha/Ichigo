import numpy as np
from classifier import classifier, keyboard
from math import ceil, exp, log
from random import shuffle, random
from collections import defaultdict


### INIT ###

keyboard_chars = "qwertyuiopasdfghjkl'zxcvbnm,.-QWERTYUIOPASDFGHJKL\"ZXCVBNM<>_"

# Penalties
tg_p = [
    1.0,
    1.0,
    1.0,
    26.29433504903491,
    44.34070607270727,
    39.985964488415064,
    -12.659238662655303,
    -10.962297486109717,
    -4.275039544251013,
    -8.232024616128134,
    23.979515076441263,
    39.81279559140114,
    1.0,
    1.0,
    1.0,
    1.0,
]


bg_p = [
    -1.358643503656903,
    -234.1035561914599,
    28.25054846797392,
    3.6912380965766274,
    2.874734639838716,
    2.747640741693443,
    1.0000000008308518,
    1.0000000001107183,
    0.9999999998832696,
    0.9999999991680705,
    0.9999999996838225,
    1.0000000000469154,
    1.0000000003742693,
    1.0000000003973035,
    0.999999999300908,
    6.513187936250937,
    1.784077376888822,
    2.6557542158071836,
    4.915445765911138,
    0.07916350666219463,
    3.980729394433583,
    3.0790757069784127,
    0.8941207401915743,
    13.883917419971066,
    3.014361043265274,
    5.842606937431901,
    11.884473422363358,
    12.475971166710973,
    -0.26102045297438203,
    3.0000041425533843,
    10.17570637719123,
    0.9999999989144547,
    0.9999999982788965,
    0.9999999995313589,
    1.0000000003764393,
    1.000000000054269,
    1.000000000252933,
    1.0000000003951657,
    1.0000000006756382,
    0.9999999993676232,
    0.9999999995657604,
    1.0000000003895777,
    1.0000000008985634,
    0.1400841930374359,
    -14.033378907821941,
    1.8026476246863148,
]


# getting trigrams and their frequencies
trigrams, tg_freqs = [], []
tg_percentages = {}
tg_coverage = 90  # the percentage of tg's to use

with open("trigrams.txt") as f:
    valid_c = "qwertyuiopasdfghjkl'zxcvbnm,.-"

    for trigram, freq in (l.split("\t") for l in f):
        if all([c in valid_c for c in trigram]):
            trigrams.append(trigram)
            tg_freqs.append(int(freq))

    percentages = [0] * 100
    total_count = sum(tg_freqs)
    elapsed = 0

    for i in range(len(tg_freqs)):
        percentage = int(100 * (elapsed / total_count))
        tg_percentages[percentage + 1] = i
        elapsed += tg_freqs[i]

# trimming our tg data to the amount of data we'll actually be processing
tg_freqs = np.array(tg_freqs[: tg_percentages[tg_coverage]])
trigrams = trigrams[: tg_percentages[tg_coverage]]
print("Processed trigram data")

# getting bigrams and their frequencies and storing it as a dict
bigram_to_freq = defaultdict(int)

with open("bigrams.txt") as f:
    for k, v in (l.split("\t") for l in f):
        bigram_to_freq[k] = int(v)

print("Processed bigram data")


class optimizer:
    def __init__(self):
        self.t0 = 0
        self.cooling_schedule = "default"
        self.keyboard = keyboard(keyboard_chars)
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
        print("getting initial temperature")

        # An initial guess for t1
        tn = 2_000_000_000
        acceptance_probability = 0

        # Repeat guess
        while abs(acceptance_probability - x0) > epsilon:
            energies = []

            # test all possible swaps
            for i, k1 in enumerate(self.keyboard.lowercase[:-1]):
                for k2 in self.keyboard.lowercase[i + 1 :]:
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

        print(f"initial temperature found, t1 = {tn}")

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

        self.fitness = int(np.sum(self.get_trigram_times() * tg_freqs))
        # print((100_000_000 / (self.fitness / 1000 / 60)))
        # self.new_bg_scores[bg] = predicted_time * freq
        # delta = self.new_bg_scores[bg] - self.bg_scores[bg]

        # self.fitness += delta

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
            print(int(self.fitness))
            print(self.keyboard)

    def get_bg_features(self, bg):
        ((ax, ay), (bx, by)) = [self.keyboard.get_pos(c) for c in bg]

        freq = bigram_to_freq[bg]

        caps1 = bg[0] in self.keyboard.uppercase
        caps2 = bg[1] in self.keyboard.uppercase

        # Row features
        space1 = ay == 0
        space2 = by == 0
        bottom1 = ay == 1
        bottom2 = by == 1
        home2 = by == 2
        top1 = ay == 3
        top2 = by == 3

        # Column features
        pinky1 = abs(ax) == 5
        pinky2 = abs(bx) == 5
        ring1 = abs(ax) == 4
        ring2 = abs(bx) == 4
        middle1 = abs(ax) == 3
        middle2 = abs(bx) == 3
        index1 = abs(ax) in (1, 2)
        index2 = abs(bx) in (1, 2)

        row_offsets = [0.5, 0, -0.25]

        dy = abs(ay - by)
        dx = abs((ax + row_offsets[ay - 1]) - (bx + row_offsets[by - 1]))

        lateral = abs(bx) == 1

        # Classifications

        # A scissor is a bigram with a Δy==2, and a row-stagger Δx <= 1
        # or a bigram where the long finger curls and the short finger stretches 'xq'
        scissor = dy == 2 and dx <= 1

        # middle and ring don't like curling, index and pinky don't like to stretch
        scissor |= (pinky1 and top1 and ring2 and bottom2) or (
            ring1 and bottom1 and pinky2 and top2
        )
        scissor |= (ring1 and top1 and middle2 and bottom2) or (
            middle1 and bottom1 and ring2 and top2
        )
        scissor |= (index1 and top1 and middle2 and bottom2) or (
            middle1 and bottom1 and index1 and top1
        )

        shb = False

        if not (space1 or space2):
            shb = (ax // abs(ax)) == (bx // abs(bx))

        sfb = (ax == bx) or (shb and (abs(ax) in (1, 2) and abs(bx) in (1, 2)))
        lsb = shb * ((index1 and middle2) | (index2 and middle1) and dx > 1.5)

        return (
            freq,
            space1,
            space2,
            bottom2,
            home2,
            top2,
            pinky1,
            pinky2,
            ring1,
            ring2,
            middle1,
            middle2,
            index1,
            index2,
            shb,
            sfb,
            scissor,
            lsb,
            lateral,
            caps1,
            caps2,
            dx,
            dy,
        )

    def get_bg_time(self, bg):
        (
            bg_freq,
            bg_space1,
            bg_space2,
            bg_bottom2,
            bg_home2,
            bg_top2,
            bg_pinky1,
            bg_pinky2,
            bg_ring1,
            bg_ring2,
            bg_middle1,
            bg_middle2,
            bg_index1,
            bg_index2,
            bg_shb,
            bg_sfb,
            bg_scissor,
            bg_lsb,
            bg_lateral,
            bg_caps1,
            bg_caps2,
            bg_dx,
            bg_dy,
        ) = self.get_bg_features(bg)

        freq_pen = bg_p[0] * np.log(bg_freq + bg_p[1]) + bg_p[2]

        # Row penalties
        base_row_pen = bg_p[3] * bg_bottom2 + bg_p[4] * bg_home2 + bg_p[5] * bg_top2

        shb_row_pen = bg_p[6] * bg_bottom2 + bg_p[7] * bg_home2 + bg_p[8] * bg_top2
        alt_row_pen = bg_p[9] * bg_bottom2 + bg_p[10] * bg_home2 + bg_p[11] * bg_top2
        sfb_row_pen = bg_p[12] * bg_bottom2 + bg_p[13] * bg_home2 + bg_p[14] * bg_top2

        # Finger penalties
        sfb_finger_pen = (
            bg_p[15] * bg_pinky2
            + bg_p[16] * bg_ring2
            + bg_p[17] * bg_middle2
            + bg_p[18] * bg_index2
        )
        base_finger_pen = (
            bg_p[19] * bg_pinky2
            + bg_p[20] * bg_ring2
            + bg_p[21] * bg_middle2
            + bg_p[22] * bg_index2
        )
        shb_finger_pen = (
            bg_p[23] * bg_pinky2
            + bg_p[24] * bg_ring2
            + bg_p[25] * bg_middle2
            + bg_p[26] * bg_index2
        )
        alt_finger_pen = (
            bg_p[27] * bg_pinky2
            + bg_p[28] * bg_ring2
            + bg_p[29] * bg_middle2
            + bg_p[30] * bg_index2
        )
        shift_finger_pen1 = (
            bg_p[31] * bg_pinky1
            + bg_p[32] * bg_ring1
            + bg_p[33] * bg_middle1
            + bg_p[34] * bg_index1
        )
        shift_finger_pen2 = (
            bg_p[35] * bg_pinky2
            + bg_p[36] * bg_ring2
            + bg_p[37] * bg_middle2
            + bg_p[38] * bg_index2
        )

        # Aggregate penalties for classes
        shb_pen = shb_finger_pen  # * (shb_row_pen)
        alt_pen = alt_finger_pen  # * (alt_row_pen)
        sfb_pen = sfb_finger_pen  # + (sfb_row_pen)

        # class penalties
        base_weight = (
            1
            + (base_row_pen) * (base_finger_pen)
            # + (bg_space2 * bg_p[39] + bg_space1 * bg_p[40])
            + bg_p[43] * bg_lateral
        )
        # base_weight *= (bg_caps1 * shift_finger_pen1 + bg_p[41]) * (
        #     bg_caps2 * shift_finger_pen2 + bg_p[42]
        # )
        shb_weight = (
            (bg_shb * (1 - bg_sfb))
            * (shb_pen)  # + bg_p[43] * bg_lsb
            # * (bg_scissor + bg_p[34])
            # * (bg_p[40] * bg_dy + bg_p[41])
        )
        sigmoid = 1 / (1 + exp(-bg_p[35]))
        sfb_weight = bg_sfb * sfb_pen * ((bg_dx**2 + bg_dy**2) ** sigmoid + bg_p[31])
        alt_weight = (1 - bg_shb) * alt_pen

        return freq_pen * (base_weight + alt_weight + shb_weight + sfb_weight)

    def get_tg_features(self):
        data_size = len(trigrams)

        tg_bg1_prediction = np.zeros(data_size)
        tg_bg2_prediction = np.zeros(data_size)
        tg_redirect = np.zeros(data_size)
        tg_bad = np.zeros(data_size)
        sg_bottom = np.zeros(data_size)
        sg_home = np.zeros(data_size)
        sg_top = np.zeros(data_size)
        sg_pinky = np.zeros(data_size)
        sg_ring = np.zeros(data_size)
        sg_middle = np.zeros(data_size)
        sg_index = np.zeros(data_size)
        sg_sfs = np.zeros(data_size)

        for i in range(data_size):
            tg = trigrams[i]

            # extracting position
            ((ax, ay), (bx, by), (cx, cy)) = [self.keyboard.get_pos(c) for c in tg]

            # getting bigram times
            tg_bg1_prediction[i] = self.get_bg_time(tg[:2])
            tg_bg2_prediction[i] = self.get_bg_time(tg[1:])

            # redirect penalty
            sht = False

            if 0 not in (ax, bx, cx):
                sht = ax / abs(ax) == bx / abs(bx) == cx / abs(cx)

            if sht:
                tg_redirect[i] = (abs(ax) < abs(bx) and abs(cx) < abs(bx)) or (
                    abs(ax) > abs(bx) and abs(cx) > abs(bx)
                )
                tg_bad[i] = tg_redirect[i] * any(
                    [x in (1, 2) for x in (abs(ax), abs(bx), abs(cx))]
                )
                tg_redirect[i] *= 1 - tg_bad[i]

            # Skipgram Penalties

            # row features
            sg_bottom[i] = cy == 1
            sg_home[i] = cy == 2
            sg_top[i] = cy == 3

            # Column features
            sg_pinky[i] = abs(cx) == 5
            sg_ring[i] = abs(cx) == 4
            sg_middle[i] = abs(cx) == 3
            sg_index[i] = abs(cx) in (1, 2)

            # SFS
            if ax != 0 and cx != 0:
                sg_sfs[i] = (ax / abs(ax)) == (cx / abs(cx))
                sg_sfs[i] *= (abs(ax) == abs(cx)) | (abs(ax) in (1, 2)) and sg_index[i]

        return (
            tg_bg1_prediction,
            tg_bg2_prediction,
            tg_redirect,
            tg_bad,
            sg_bottom,
            sg_home,
            sg_top,
            sg_pinky,
            sg_ring,
            sg_middle,
            sg_index,
            sg_sfs,
        )

    def get_trigram_times(self):
        (
            tg_bg1_prediction,
            tg_bg2_prediction,
            tg_redirect,
            tg_bad,
            sg_bottom,
            sg_home,
            sg_top,
            sg_pinky,
            sg_ring,
            sg_middle,
            sg_index,
            sg_sfs,
        ) = self.get_tg_features()

        # freq_pen = tg_p[0] * np.log(tg_freqs + tg_p[1]) + tg_p[2]

        # finger penalty
        sfs_row_pen = tg_p[3] * sg_bottom + tg_p[4] * sg_home + tg_p[5] * sg_top[2]
        sfs_finger_pen = (
            tg_p[6] * sg_pinky
            + tg_p[7] * sg_ring
            + tg_p[8] * sg_middle
            + tg_p[9] * sg_index
        )

        # sfs penalty
        sfs_weight = sg_sfs * (sfs_row_pen + sfs_finger_pen)

        return (
            # freq_pen *
            tg_bg1_prediction
            + tg_bg2_prediction
            + sfs_weight
            + (tg_p[10] * tg_redirect)
            + (tg_p[11] * tg_bad)
        )


best_keeb = None
best_score = float("inf")

for _ in range(10):
    o = optimizer()
    o.optimize()

    print("Fitness", int(o.fitness))

    if o.fitness < best_score:
        best_score = o.fitness
        print("new best")
        best_keeb = o.keyboard

print("best score")
print(best_keeb)
