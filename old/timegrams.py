import concurrent.futures
import pandas as pd
import re
import os
from collections import defaultdict
from mapper import mappings

bigrams_freqs = {
    k: int(v) for (k, v) in [l.strip().split("\t") for l in open("ngrams/bigrams.txt")]
}

debug = False


def print_debug(*args):
    if debug:
        print(*args)


f = open("wpm_metadata.txt")
session_wpms = dict(map(lambda x: map(int, x.split(" ")), [l for l in f]))
f.close()

DATA_TYPES = {
    # "bistrokes": (2, 0),
    "tristrokes": (3, 0),
    "1-skip": (2, 1),
}

valid_chars = set("qwertyuiopasdfghjkl;zxcvbnm,./QWERTYUIOPASDFGHJKL:ZXCVBNM<>?")


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
            duration = int(float(stroke_data[-1][1]) - float(stroke_data[0][1]))
            stroke = "".join([l[0] for l in stroke_data])

            if len(stroke) != size:
                print(file)
                print(stroke)
                print(i)

            if all([c in valid_chars for c in stroke]) and len(stroke) == size:
                strokes[stroke].append((wpm, duration))


def process_data_type(alias, size, skip, wpm, shared_strokes):
    try:
        with open(f"nstrokes/{alias}.txt", "w") as output:
            for layout in ("azerty", "dvorak", "qwerty", "qwertz"):
                strokes = shared_strokes[alias]
                participants = pd.read_csv("meta/metadata_participants.txt", sep="\t")

                for i, p in participants[
                    (participants["FINGERS"] == "9-10")
                    & (participants["KEYBOARD_TYPE"] != "on-screen")
                    & (participants["LAYOUT"] == layout)
                ].iterrows():
                    ID = p["PARTICIPANT_ID"]
                    participant_dir = "typingrecords/" + str(ID).zfill(6)

                    for file_name in os.listdir(participant_dir):
                        match = re.match(r"(.*)_processed\.txt", file_name)

                        if (
                            match
                            and match.group(1).isdigit()
                            and session_wpms[int(match.group(1))] > 0
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

                output_lines = []

                for k in sorted(strokes.keys()):
                    freq = str(bigrams_freqs.get(mappings[layout].decode_str(k), 0))
                    output_lines.append(
                        [layout, k, freq, *map(str, sorted(strokes[k]))]
                    )

                for l in sorted(output_lines, key=lambda x: int(x[2]), reverse=True):
                    output.write("\t".join(l) + "\n")

                # resetting the dict for the next layout
                shared_strokes[alias] = defaultdict(list)

        print(f"nstrokes/{alias}.txt")
    except Exception as e:
        print(e)


def get_strokes(wpm):
    shared_strokes = {alias: defaultdict(list) for alias in DATA_TYPES.keys()}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_data_type, alias, size, skip, wpm, shared_strokes)
            for alias, (size, skip) in DATA_TYPES.items()
        ]
        concurrent.futures.wait(futures)


wpms = [int(l.split(" ")[1]) for l in open("wpm_metadata.txt")]
avg_wpm = int(sum(wpms) / len(wpms))

for s_num in (0,):
    get_strokes(s_num)
