# 2,528,895
import pandas as pd
import re
import os

debug = True

### Layout Mapping ###
shifted_keys = 'QWERTYUIOPASDFGHJKLZXCVBNM,.<>?:"{}~!@\#$%^&*()_+|'

# to handle javascript keycodes, because some files don't include the characters, but still have keycodes
with open("keycodes.tsv") as f:
    keycode_to_char = {l.split("\t")[0]: l.strip("\n").split("\t")[1] for l in f}


class key_record:
    def __init__(self, char, start_time, is_correct):
        self.char = char
        self.start_time = start_time
        self.is_correct = is_correct

    def copy(self):
        return key_record(self.char, self.start_time, self.is_correct)

    def __repr__(self):
        return f"{self.char} {self.start_time} {self.is_correct}"

    def __str__(self):
        return f"{self.char}, {self.start_time}, {self.is_correct}"


def print_debug(*args):
    if debug:
        print(*args)


# to handle the fact that some files are broken... in multiple ways! Some key codes are wrong, so we can't rely on those, but some files don't have all
# keys typed, instead it has "" so we have to check
def normalize_key(lines, i):
    if lines[i][-2] == "":
        return keycode_to_char[lines[i][-1].strip("\n")]
    return lines[i][-2]


#### Typo Finder ####
# Build an index mapping elements of sequence_b to their indices
def build_index(sequence_b):
    # Construct a dictionary where keys are elements of sequence_b and values are lists of their indices in sequence_b
    element_to_indices = {}

    for index, element in enumerate(sequence_b):
        indices = element_to_indices.setdefault(element, [])
        indices.append(index)

    return element_to_indices


# Find the longest matching str between two slices of the sequences
def find_longest_match(
    sequence_a, sequence_b, element_to_indices, start_a, end_a, start_b, end_b
):
    # Initialize variables to track the best matching str
    best_start_a, best_start_b, best_size = start_a, start_b, 0
    j_to_length = {}
    empty_list = []

    # Iterate over elements in sequence a to find the longest matching str
    for i in range(start_a, end_a):
        get_j_to_length = j_to_length.get
        new_j_to_length = {}

        for j in element_to_indices.get(sequence_a[i], empty_list):
            if j < start_b:
                continue

            if j >= end_b:
                break

            length = new_j_to_length[j] = get_j_to_length(j - 1, 0) + 1

            # Update the best matching str if a longer one is found
            if length > best_size:
                best_start_a, best_start_b, best_size = (
                    i - length + 1,
                    j - length + 1,
                    length,
                )

        j_to_length = new_j_to_length

    # Extend the matching str with junk elements if applicable
    while (
        best_start_a > start_a
        and best_start_b > start_b
        and sequence_a[best_start_a - 1] == sequence_b[best_start_b - 1]
    ):
        best_start_a, best_start_b, best_size = (
            best_start_a - 1,
            best_start_b - 1,
            best_size + 1,
        )

    while (
        best_start_a + best_size < end_a
        and best_start_b + best_size < end_b
        and sequence_a[best_start_a + best_size] == sequence_b[best_start_b + best_size]
    ):
        best_size += 1

    # Return the Match tuple representing the longest matching str
    return (best_start_a, best_start_b, best_size)


# Get list of matching strs between sequences
def get_matching_strings(str_a, str_b):
    # Maintain a queue of the upper and lower bounds of the str a and the str b
    len_a, len_b = len(str_a), len(str_b)
    queue = [(0, len_a, 0, len_b)]
    matching_strs = []

    element_to_indices = build_index(str_b)

    # Loop until the queue is empty, finding matching strs
    while queue:
        start_a, end_a, start_b, end_b = queue.pop()
        i, j, size = x = find_longest_match(
            str_a, str_b, element_to_indices, start_a, end_a, start_b, end_b
        )

        # Add the matching str to the list
        if size:
            matching_strs.append(x)

            # Add new subregions to the queue for further examination
            if start_a < i and start_b < j:
                queue.append((start_a, i, start_b, j))

            if i + size < end_a and j + size < end_b:
                queue.append((i + size, end_a, j + size, end_b))

    # Sort the matching strs and merge adjacent strs
    matching_strs.sort()
    prev_start_a = prev_start_b = prev_size = 0
    non_adjacent = []

    for cur_start_a, cur_start_b, cur_size in matching_strs:
        if (
            prev_start_a + prev_size == cur_start_a
            and prev_start_b + prev_size == cur_start_b
        ):
            prev_size += cur_size
        else:
            if prev_size:
                non_adjacent.append((prev_start_a, prev_start_b, prev_size))

            prev_start_a, prev_start_b, prev_size = cur_start_a, cur_start_b, cur_size

    if prev_size:
        non_adjacent.append((prev_start_a, prev_start_b, prev_size))

    return [tuple(item) for item in non_adjacent]


### Processing and Writing the Output ####
def get_duration(data):
    start, end = map(float, data[5:7])
    return end - start


def amend_key_record(correct_str, typing_record, record, threshold=3):
    typed_str = "".join([a for a, b in typing_record])
    matching_strs = get_matching_strings(typed_str, correct_str)

    for i, (src, _, size) in enumerate(matching_strs):
        if size >= threshold:
            for j in range(size):
                typing_record[src + j][1].is_correct = True

            # Adding a duplicate for when a character was missed
            if i < len(matching_strs) - 1:
                incorrect_record = typing_record[src + size][1]
                new_record = incorrect_record.copy()
                new_record.is_correct = False
                record_i = record.index(incorrect_record)

                record = record[:record_i] + [new_record] + record[record_i:]

    return record


def is_capital_pair(i, lines):
    return (normalize_key(lines, i) == "SHIFT") or (
        normalize_key(lines, i) == "CAPS_LOCK"
    )


def calculate_wpm(correct_chars_typed, total_duration):
    return round((correct_chars_typed / 5) / (total_duration / 60000))


def process_typing_session(session_file, wpm_file):
    lines = [l.split("\t") for l in session_file]
    correct_string = lines[0][2]
    key_records = []
    curr_string = []
    total_duration = int(float(lines[-1][5]) - float(lines[0][5]))
    is_time_shifted = False
    new_start = 0

    for i, line in enumerate(lines):
        start_time = line[5]
        char = normalize_key(lines, i)

        # backspaces are inconsistent in some files and can either represent one or more backspaces
        if len(char) > 1:
            if is_capital_pair(i, lines):
                if not is_time_shifted:
                    is_time_shifted = True
                    new_start = start_time
                continue

            if is_time_shifted:
                start_time = new_start
                is_time_shifted = False

            key_records = amend_key_record(correct_string, curr_string, key_records)
            curr_string = []
            key_records.append(key_record(char, start_time, False))
            continue

        if is_time_shifted:
            start_time = new_start
            is_time_shifted = False

        key_records.append(key_record(char, start_time, False))
        curr_string.append((char, key_records[-1]))

    # finally amendment now that the processing has completed
    key_records = amend_key_record(correct_string, curr_string, key_records)

    # Write the processed session file
    match = re.match(r".*/(.*)/(.*)\.txt", session_file.name)
    participant_id = match.group(1)

    if not os.path.exists(f"session_data/{participant_id}"):
        os.makedirs(f"session_data/{participant_id}")

    print(participant_id)
    session_id = match.group(2)

    print_debug(f"session_data/{participant_id}/{session_id}_processed.txt")
    with open(
        f"session_data/{participant_id}/{session_id}_processed.txt", "w"
    ) as new_file:
        for k in key_records:
            new_file.write(str(k) + "\n")

    # process and save wpm metadata
    correct_typed = sum([int(k.is_correct) for k in key_records])
    wpm = calculate_wpm(correct_typed, total_duration)
    print_debug("wpm", wpm)

    wpm_file.write(f"{session_id} {wpm}\n")


### Bringing it all together baby :))
participants = pd.read_csv("meta/metadata_participants.txt", sep="\t")


# participants["AVG_WPM_15"] > 0
# 168161/1825516
with open("wpm_metadata.txt", "w") as wpm_record:
    for i, p in participants.iterrows():
        ID = p["PARTICIPANT_ID"]
        participant_dir = "typingrecords/" + str(ID).zfill(6)
        print(p["LAYOUT"])

        for file_name in os.listdir(participant_dir):
            if re.match(r"(.*)\.txt", file_name).group(1).isdigit():
                print_debug(participant_dir + "/" + file_name)

                with open(participant_dir + "/" + file_name) as file:
                    process_typing_session(file, wpm_record)
