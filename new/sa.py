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
    1.0000000102102182,
    -2.1016308706867277,
    10.313611463019928,
    7.690132310321646,
    3.6046218804847463,
    -2.40775104239068,
    2.7971512270644574,
    1.4346995043031492,
    14.841662543111768,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]


bg_p = [
    -10.300437125600473,
    -1642.9346782045966,
    250.29055874449554,
    1.0950990814078618,
    -1.6887582824251037,
    1.000000019547364,
    -0.26598860617770287,
    1.0681331348399665,
    0.9999999834871871,
    -0.2851032381361033,
    0.19817726955962967,
    0.9999999611808169,
    -0.3121306382645282,
    -0.28403284008704166,
    0.9999999999389337,
    0.3238312010352593,
    0.336672986553846,
    0.33218545046838965,
    0.3236488928048248,
    0.02421578484708868,
    0.15249404511102613,
    0.1214709042010299,
    0.12211245588015388,
    -0.06988203056350295,
    0.6116903979996243,
    0.5900425571498125,
    0.2626787173607642,
    0.4088201158750971,
    1.152372145539701,
    0.9605459343034942,
    0.8287624123864489,
    0.9999999585951099,
    1.0000000607724768,
    1.0000000136653995,
    1.0000000341101225,
    1.0000000409788892,
    0.9999999743087522,
    0.9999999731524938,
    0.9999999410759218,
    1.0000000382887186,
    0.9999999633782,
    0.9999999955633581,
    1.000000008442152,
    0.008040665843636463,
    0.9999999552424983,
    9.271723577427458,
]


# getting trigrams and their frequencies
trigrams, tg_freqs = [], []
tg_percentages = {}
tg_coverage = 95  # the percentage of tg's to use

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
bg_times = {}
print("Processed trigram data")


# trigram penalties
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

                if delta < 0:
                    self.accept()
                    stays = 0

                # Metropolis criterion
                if delta > 0:
                    if random() < exp(-delta / self.temp):
                        self.accept()
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

        freq_pen = (
            bg_p[0] * np.log(np.clip(bg_freq + bg_p[1], a_min=1, a_max=None)) + bg_p[2]
        )

        # Row penalties
        base_row_pen = bg_p[3] * (bg_home2 + bg_top2) + bg_p[4] * bg_bottom2
        shb_row_pen = bg_p[6] * (bg_home2 + bg_top2) + bg_p[7] * bg_bottom2
        alt_row_pen = bg_p[9] * (bg_home2 + bg_top2) + bg_p[10] * bg_bottom2
        sfb_row_pen = bg_p[12] * (bg_home2 + bg_top2) + bg_p[13] * bg_bottom2

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
        shb_pen = shb_finger_pen * (shb_row_pen)
        alt_pen = alt_finger_pen * (alt_row_pen)
        sfb_pen = sfb_finger_pen + (sfb_row_pen)

        # class penalties
        base_weight = (
            1
            + (base_row_pen * base_finger_pen)
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
        # sigmoid = 1 / (1 + exp(-bg_p[35]))
        sfb_weight = bg_sfb * sfb_pen * ((bg_dx**2 + bg_dy**2) ** 1 + bg_p[45])
        alt_weight = (1 - bg_shb) * alt_pen

        return freq_pen * (base_weight + alt_weight + shb_weight + sfb_weight)

    def get_tg_features(self):

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
            + tg_p[10]
            # + (tg_p[10] * tg_redirect)
            # + (tg_p[11] * tg_bad)
        )


best_keeb = None
best_score = float("inf")

for _ in range(5):
    o = optimizer()
    o.optimize()

    print("Fitness", int(o.fitness))

    if o.fitness < best_score:
        best_score = o.fitness
        print("new best")
        best_keeb = o.keyboard

print("best score")
print(best_keeb)
