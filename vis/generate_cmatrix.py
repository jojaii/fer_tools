# TODO: entry class
# class attributes are: frame, fps, and sentiment_probs

# each filename is a label
directory = "/home/ubuntu/"
# original label arrangement
label = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
files = ["angry.txt", "disgust.txt", "fear.txt", "happy.txt", "sad.txt", "surprise.txt", "neutral.txt"]


# initialize entry list
entry_list = []
# actual values in confusion matrix
y_actual = []

for file in files:
    open_file = open(file, "r")

    lines = open_file.read().splitlines()
    frame_count = len(lines)

    # get index of current opened file
    correct_label = files.index(file)
    # populate array with correct labels according to number of frames in the file
    # (represented by frame_count)
    y_actual.append(labels[correct_label] * frame_count)


    for line in lines:
        temp = line.split(" ")
        entry = Entry()
        # TODO: how to separate different frame numbers of different files??
        entry.frame_number = temp[0]
        entry.fps = temp[1]
        # extract first to last sentiment probability
        entry.sentiment_probs = temp[2:len(temp)]
        entry_list.append(entry)
    open_file.close()

