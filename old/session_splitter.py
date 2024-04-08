import pandas as pd
import os

# This decomposed the participant test files into folders that have each test they completed
participants = pd.read_csv("metadata_participants.txt", sep="\t")

for ID in participants["PARTICIPANT_ID"]:
    participant_file_name = "files/" + str(ID).zfill(6)

    if not os.path.exists(participant_file_name + ".txt"):
        print(participant_file_name + ".txt", "does not exist")
        continue

    print(participant_file_name + ".txt")

    # make a folder to session files in, if it already exists, we don't need to do new work
    if not os.path.exists(participant_file_name):
        os.makedirs(participant_file_name)

        session_id = ""
        session_file = None

        with open(participant_file_name + ".txt", encoding="latin-1") as file:
            # skip header
            file.readline()
            lines_remain = True

            while lines_remain:
                l = file.readline()

                if not l:
                    break

                data = l.split("\t")

                # handeling the fact that lines can have \n as a valid input type
                if len(data) < 9:
                    data += ["ENTER", "77"]
                    file.readline()

                # make a new file for each session
                if session_id != data[1]:
                    session_id = data[1]
                    session_file = open(
                        f"{participant_file_name}/{session_id}.txt", "w"
                    )

                    if "\n" in session_id:
                        raise ValueError(f"!{participant_file_name},{session_id}")

                # super jank fix because they included tabs as a data point :face_palm:
                if len(data) > 9:
                    data = data[:7] + ["TAB"] + [data[-1]]

                session_file.write("\t".join(data))
