import numpy as np
from classifier import classifier, keyboard
from math import ceil, exp, log
from random import shuffle, random
from collections import defaultdict

# from LED import *
# set_orientation(1)


### INIT ###

# keyboard_chars = 'qwertyuiopasdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL"ZXCVBNM<>_'
keyboard_chars = "qwertyuiopasdfghjkl'zxcvbnm,.-QWERTYUIOPASDFGHJKL\"ZXCVBNM<>_"
# keyboard_chars = "yadcgkhur.sotpfmlenij'qbwvz-x,QWERTYUIOPASDFGHJKL\"ZXCVBNM<>_"

initial_temp = None

# Penalties
bg_p = [
    -13.006290200857604,
    -1110.6975118720209,
    301.79595431794627,
    0.2373529046003713,
    -0.3483169122234296,
    -0.44869126126093584,
    0.5183895781213851,
    -0.3675121927036903,
    0.21114436760532884,
    -0.11053904996244854,
    0.4969806603740193,
    0.2270610038405438,
    0.24377147190687995,
    0.24058025690698362,
    0.18606952101779842,
    0.13566910965517368,
    0.4269095628160329,
    1.0659786791611865,
    0.9072922405602595,
    0.8267421145473832,
    0.683055916507772,
    0.4223096257759204,
    0.9144166084516641,
    1.0013425051520004,
    0.5423592762848283,
    0.3124915691983608,
    0.7636142108419756,
    1.246192084201704,
    1.0748870559594512,
    0.9737199188685832,
    0.761229963934242,
    2.932362314618125,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]


tg_p = [
    1.0,
    1.0,
    -30.949166125789187,
    10.49968914570526,
    1.3812508525050837,
    3.557707209342576,
    -1.5392605080726125,
    -1.3101250448185828,
    -11.732288602506747,
    1.8697907379041518,
    19.60346057380557,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]

# getting trigrams and their frequencies
trigrams, tg_freqs = [], []
row_offsets = [0.5, 0, -0.25]
tg_percentages = {}
tg_coverage = 100  # the percentage of tg's to use

with open("trigrams.txt") as f:
    # valid_c = "qwertyuiopasdfghjkl;zxcvbnm,./"
    valid_c = "qwertyuiopasdfghjkl-zxcvbnm,.'"

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
sg_lateral = np.zeros(data_size)
sg_dx = np.zeros(data_size)
sg_dy = np.zeros(data_size)
sg_sfs = np.zeros(data_size)

# getting bigrams and their frequencies and storing it as a dict
bigram_to_freq = defaultdict(int)
bigram_times = {}

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
        self.affected_indices = range(data_size)
        self.fitness = 0
        self.prev_fitness = 0
        self.a = 0.995
        self.bg_times = {}

        self.get_fitness()

        print(self.keyboard)
        print(self.fitness)

        # Test real quick
        l1 = "yadcgkhur.sotpfmlenij'qbwvz-x,QWERTYUIOPASDFGHJKL\"ZXCVBNM<>_"
        l2 = "yrdcgkhua.sntpfmleoijxqbwvz-',QWERTYUIOPASDFGHJKL\"ZXCVBNM<>_"

        for k1, k2 in ((l1, l2), (l2, l1)):
            self.keyboard = keyboard(k1)

            t1 = self.update_trigram_times() * tg_freqs

            self.keyboard = keyboard(k2)

            t2 = self.update_trigram_times() * tg_freqs

            self.get_fitness()
            print("------------------------------------")
            print(self.keyboard)
            print(self.fitness)
            print()

            total = 0

            for i in list(np.argsort(-np.abs(t1 - t2))[np.where(t1 != t2)])[:10]:
                bg_t = int(tg_bg1_prediction[i]) + int(tg_bg2_prediction[i])
                print(
                    trigrams[i],
                    f"{bg_t} = {int(tg_bg1_prediction[i])} + {int(tg_bg2_prediction[i])}",
                )

                total += bg_t

                # print(
                #     trigrams[i][:2],
                #     *map(int, self.get_bg_features(trigrams[i][:2])[14:]),
                # )
                # print(
                #     trigrams[i][1:],
                #     *map(int, self.get_bg_features(trigrams[i][1:])[14:]),
                # )
            print(total)

        print(1 / 0)

        self.accept()

        if initial_temp == None:
            self.temp = self.get_initial_temperature(0.7)
        else:
            self.temp = initial_temp
        self.stopping_point = self.get_stopping_point()

    def swap(self, k1=None, k2=None):
        if k1 != None and k2 != None:
            self.keyboard.swap(k1, k2)
        else:
            self.keyboard.random_swap()

        self.affected_indices = [
            i
            for i, tg in enumerate(trigrams)
            if any([c in self.keyboard.swap_pair for c in tg])
        ]

        self.get_fitness()

    def debug_draw(self):
        """refresh()
        set_font(FNT_LARGE)
        s = repr(self.keyboard).replace(" ", "").split("\n")

        center_text()
        draw_text(get_width_adjusted() // 2, get_height_adjusted() // 2 + 6, s[1], CYAN)

        align_text_bottom()
        draw_text(get_width_adjusted() // 2, get_height_adjusted() // 2, s[0], CYAN)

        align_text_top()

        if initial_temp == None:
            draw_text(get_width_adjusted() // 2, 0, "initializing", CYAN)
        else:
            draw_text(get_width_adjusted() // 2, 0, f"{self.fitness}", CYAN)

        draw_text(
            get_width_adjusted() // 2, get_height_adjusted() // 2 + 12, s[2], CYAN
        )

        # draw_text()
        draw()"""

    def accept(self):
        # self.bg_scores.update(self.new_bg_scores)
        pass

    def reject(self):
        self.keyboard.undo_swap()
        self.update_trigram_times()
        self.fitness = self.prev_fitness

    def get_initial_temperature(self, x0, epsilon=0.01):
        global initial_temp
        print("getting initial temperature")

        # An initial guess for t1
        tn = 3_000_000_000
        acceptance_probability = 0

        # Repeat guess
        while abs(acceptance_probability - x0) > epsilon:
            energies = []

            # test all possible swaps
            for i, k1 in enumerate(self.keyboard.lowercase[:-1]):
                for k2 in self.keyboard.lowercase[i + 1 :]:
                    self.swap(k1, k2)

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
        initial_temp = tn

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

        # print("calculating fitness")
        # print(trigrams[:10])
        # print(self.update_trigram_times()[:10])
        self.fitness = int(np.sum(self.update_trigram_times() * tg_freqs))

        # print("done")
        # print((100_000_000 / (self.fitness / 1000 / 60)))
        # self.new_bg_scores[bg] = predicted_time * freq
        # delta = self.new_bg_scores[bg] - self.bg_scores[bg]

        # self.fitness += delta

    def optimize(self):
        stays = 0

        while stays < self.stopping_point:
            # markov chain
            for _ in range(30):
                self.swap()

                delta = self.fitness - self.prev_fitness

                if delta < 0:
                    self.accept()
                    stays = 0

                # Metropolis criterion
                if delta > 0:
                    if random() < exp(-delta / self.temp):
                        self.accept()
                        stays -= 1
                    else:
                        self.reject()
                        stays += 1

            self.cool()
            print(self.fitness, f"@{tg_coverage}% a={self.a}")
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
        if bg not in self.bg_times:
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
                bg_p[0] * np.log(np.clip(bg_freq + bg_p[1], a_min=1, a_max=None))
                + bg_p[2]
            )

            # Row penalties
            base_row_pen = bg_p[3] * (bg_home2 + bg_top2) + bg_p[4] * (
                bg_top2 + bg_bottom2
            )
            shb_row_pen = bg_p[5] * (bg_home2 + bg_top2) + bg_p[6] * (
                bg_top2 + bg_bottom2
            )
            alt_row_pen = bg_p[7] * (bg_home2 + bg_top2) + bg_p[8] * (
                bg_top2 + bg_bottom2
            )
            sfb_row_pen = bg_p[9] * (bg_home2 + bg_top2) + bg_p[10] * (
                bg_top2 + bg_bottom2
            )

            # Finger penalties
            sfb_finger_pen = (
                bg_p[11] * bg_pinky2
                + bg_p[12] * bg_ring2
                + bg_p[13] * bg_middle2
                + bg_p[14] * (bg_index2 - bg_lateral)
                + bg_p[15] * bg_lateral
            )
            base_finger_pen = (
                bg_p[16] * bg_pinky2
                + bg_p[17] * bg_ring2
                + bg_p[18] * bg_middle2
                + bg_p[19] * (bg_index2 - bg_lateral)
                + bg_p[20] * bg_lateral
            )
            shb_finger_pen = (
                bg_p[21] * bg_pinky2
                + bg_p[22] * bg_ring2
                + bg_p[23] * bg_middle2
                + bg_p[24] * (bg_index2 - bg_lateral)
                + bg_p[25] * bg_lateral
            )
            alt_finger_pen = (
                bg_p[26] * bg_pinky2
                + bg_p[27] * bg_ring2
                + bg_p[28] * bg_middle2
                + bg_p[29] * (bg_index2 - bg_lateral)
                + bg_p[30] * bg_lateral
            )
            # shift_finger_pen1 = (
            #     bg_p[31] * bg_pinky1
            #     + bg_p[32] * bg_ring1
            #     + bg_p[33] * bg_middle1
            #     + bg_p[34] * bg_index1
            # )
            # shift_finger_pen2 = (
            #     bg_p[35] * bg_pinky2
            #     + bg_p[36] * bg_ring2
            #     + bg_p[37] * bg_middle2
            #     + bg_p[38] * bg_index2
            # )

            # Aggregate penalties for classes
            shb_pen = shb_finger_pen * (shb_row_pen)
            alt_pen = alt_finger_pen * (alt_row_pen)
            sfb_pen = sfb_finger_pen * (sfb_row_pen)

            # class penalties
            base_weight = (
                1
                + (base_row_pen * base_finger_pen)
                # + (bg_space2 * bg_p[39] + bg_space1 * bg_p[40])
                # + bg_p[43] * bg_lateral
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
            dist_pen = (bg_dx**2 + bg_dy**2) ** 0.5 + bg_p[31]
            sfb_weight = bg_sfb * sfb_pen * dist_pen
            alt_weight = (1 - bg_shb) * alt_pen

            self.bg_times[bg] = freq_pen * (
                base_weight + alt_weight + shb_weight + sfb_weight
            )

        return self.bg_times[bg]

    def update_tg_features(self):
        self.bg_times = {}

        for i in self.affected_indices:
            tg = trigrams[i]

            # extracting position
            ((ax, ay), (bx, by), (cx, cy)) = [self.keyboard.get_pos(c) for c in tg]

            # getting bigram times
            tg_bg1_prediction[i] = self.get_bg_time(tg[:2])
            tg_bg2_prediction[i] = self.get_bg_time(tg[1:])

            # redirect penalty
            sht = False

            if 0 not in (ax, bx, cx):
                sht = (ax / abs(ax)) == (bx / abs(bx)) == (cx / abs(cx))

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
            sg_lateral[i] = abs(cx) == 1

            # SFS
            if ax != 0 and cx != 0:
                sg_sfs[i] = (ax / abs(ax)) == (cx / abs(cx))
                sg_sfs[i] *= (abs(ax) == abs(cx)) or (abs(ax) in (1, 2) and sg_index[i])

            sg_dy[i] = abs(ay - by)
            sg_dx[i] = abs((ax + row_offsets[ay - 1]) - (bx + row_offsets[by - 1]))

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
            sg_lateral,
            sg_sfs,
        )

    def update_trigram_times(self):
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
            sg_lateral,
            sg_sfs,
        ) = self.update_tg_features()

        freq_pen = (
            tg_p[0] * np.log(np.clip(tg_freqs + tg_p[1], a_min=1, a_max=None)) + tg_p[2]
        )

        # finger penalty
        sfs_row_pen = tg_p[3] * (sg_top + sg_home) + tg_p[4] * (sg_top + sg_bottom)

        sfs_finger_pen = (
            tg_p[5] * sg_pinky
            + tg_p[6] * sg_ring
            + tg_p[7] * sg_middle
            + tg_p[8] * (sg_index - sg_lateral)
            + tg_p[9] * sg_lateral
        )

        # dist_pen = (sg_dx**2 + sg_dy**2) ** 0.5 + tg_p[10]

        # sfs penalty
        sfs_weight = sg_sfs * (sfs_row_pen + sfs_finger_pen)  # * dist_pen

        return (
            # freq_pen *
            tg_bg1_prediction
            + tg_bg2_prediction
            + sfs_weight
            + tg_p[10]
            #
            # + (tg_p[10] * tg_redirect)
            # + (tg_p[11] * tg_bad)
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

        with open("logfile.txt", "a") as f:
            # Write to the file
            f.write(f"{best_score} @{tg_coverage}\n")
            f.write(repr(best_keeb) + "\n")

print("best score")
print(best_keeb)
