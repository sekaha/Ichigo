import concurrent.futures
import pandas as pd
import re
import os
from collections import defaultdict
from classifier import classifier
import traceback

trigram_classifier = classifier()

ngram_freqs = {k: int(v) for (k, v) in [l.split("\t") for l in open("bigrams.txt")]}
ngram_freqs.update(
    {k: int(v) for (k, v) in [l.strip().split("\t") for l in open("trigrams.txt")]}
)

debug = True


def print_debug(*args):
    if debug:
        print(*args)


f = open("wpm_metadata.txt")
session_wpms = dict(map(lambda x: map(int, x.split(" ")), [l for l in f]))
f.close()

DATA_TYPES = {
    "bistrokes": (2, 0),
    # "tristrokes": (3, 0),
    # "1-skip": (2, 1),
}

valid_chars = set("qwertyuiopasdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>? ")


def split_lines(file):
    with open(file) as f:
        lines = [l.strip("\n").split(", ") for l in f]
    return lines


def process_window(file, size, skip, wpm, strokes, layout):
    lines = split_lines(file)
    print_debug(file)

    for i in range(len(lines) - size - skip + 1):
        window = lines[i : i + size + skip]

        if all([l[2] == "True" for l in window]):
            stroke_data = window[: int(size / 2)] + window[-int(size / 2 + 0.5) :]

            durations = [
                int(float(b) - float(a))
                for (a, b) in zip(
                    [s[1] for s in stroke_data[:-1]], [s[1] for s in stroke_data[1:]]
                )
            ]

            chars = "".join([l[0] for l in stroke_data])

            if all([c in valid_chars for c in chars]) and len(chars) == size:
                # if chars == "nq":
                #     print(file)
                #     print("special bg found")

                try:
                    strokes[
                        (
                            tuple(
                                trigram_classifier.keyboards[layout].get_pos(c)
                                for c in chars
                            ),
                            chars,
                        )
                    ].append((wpm, *durations))
                except:
                    print(f"stroke not found in {layout}")


def process_data_type(alias, size, skip, shared_strokes):
    participants = pd.read_csv("metadata_participants.txt", sep="\t")

    try:
        with open(f"{alias}.tsv", "w") as output:
            strokes = shared_strokes[alias]

            for layout in ("azerty", "dvorak", "qwerty", "qwertz"):

                for i, p in participants[
                    (participants["FINGERS"] == "9-10")
                    & (participants["KEYBOARD_TYPE"] != "on-screen")
                    & (participants["LAYOUT"] == layout)
                ].iterrows():
                    ID = p["PARTICIPANT_ID"]
                    participant_dir = "session_data/" + str(ID).zfill(6)

                    for file_name in os.listdir(participant_dir):
                        match = re.match(r"(.*)_processed\.txt", file_name)

                        if (
                            match
                            and match.group(1).isdigit()
                            # and session_wpms[int(match.group(1))] > 0
                        ):
                            print_debug(participant_dir + "/" + file_name)
                            process_window(
                                participant_dir + "/" + file_name,
                                size,
                                skip,
                                session_wpms[int(match.group(1))],
                                strokes,
                                layout,
                            )

            print(f"files processed, outputting {alias} data")
            output_lines = []

            # sort keys alphabetically and then store them in the outputlines
            for k in sorted(strokes.keys(), key=lambda x: x[1]):
                output_lines.append(
                    [
                        str(k[0]),
                        k[1],
                        *map(str, sorted(strokes[k])),
                    ]
                )

            # sort by freqeuncy and output to file
            for l in sorted(
                output_lines, key=lambda x: ngram_freqs.get(x[1], 0), reverse=True
            ):
                output.write("\t".join(l) + "\n")

            # resetting the dict for the next layout
            shared_strokes[alias] = defaultdict(list)

        print(f"{alias}.tsv")
    except Exception as e:
        traceback.print_exc()


def get_strokes():
    shared_strokes = {alias: defaultdict(list) for alias in DATA_TYPES.keys()}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_data_type, alias, size, skip, shared_strokes)
            for alias, (size, skip) in DATA_TYPES.items()
        ]
        concurrent.futures.wait(futures)


get_strokes()
