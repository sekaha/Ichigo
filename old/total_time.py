import re
import os
import pandas as pd

participants = pd.read_csv("meta/metadata_participants.txt", sep="\t")
total_duration = 0

for i, p in participants.iterrows():
    ID = p["PARTICIPANT_ID"]
    layout = p["LAYOUT"]
    participant_dir = "typingrecords/" + str(ID).zfill(6)

    for file_name in os.listdir(participant_dir):
        if re.match(r"(.*)\.txt", file_name).group(1).isdigit():
            print(participant_dir + "/" + file_name)

            with open(participant_dir + "/" + file_name) as file:
                times = [float(l.split("\t")[5]) for l in file]
                duration = times[-1] - times[0]
                total_duration += duration
                print(duration)

print(total_duration)
