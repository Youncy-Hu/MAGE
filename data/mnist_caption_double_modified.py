import os
import numpy as np
import random
import lmdb
import pickle
import platform
np.random.seed(np.random.randint(1 << 30))

num_frames = 20
seq_length = 20
image_size = 64
batch_size = 1
num_digits = 2
step_length = 0.05
digit_size = 28
frame_size = image_size ** 2


def create_reverse_dictionary(dictionary):
    dictionary_reverse = {}
    for word in dictionary:
        index = dictionary[word]
        dictionary_reverse[index] = word
    return dictionary_reverse

dictionary = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'the': 10, 'digit': 11, 'and': 12,
              'is':13, 'are':14, 'bouncing': 15, 'moving':16, 'here':17, 'there':18, 'around':19, 'jumping':20, 'up':21,
              'down':22, 'left':23, 'right':24, 'then':25, '.':26}

motion_strings = ['up', 'left', 'down', 'right', 'up then down', 'left then right', 'down then up', 'right then left']
motion_idxs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

def create_dataset():
    numbers = [i for i in range(100) if i not in [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]]
    random.shuffle(numbers)
    numbers = np.array(numbers)
    dataset = np.zeros((4, 10 * 9), dtype=np.int)
    dataset[0, :] = numbers
    dataset[1, :] = 100 + numbers
    dataset[2, :] = 200 + numbers
    dataset[3, :] = 300 + numbers
    train = []
    val = []
    count = 0
    for i in range(90):
        dummy = count % 2
        val.append(dataset[dummy, i])
        train.append(dataset[1 - dummy, i])
        count = count + 1
    for i in range(90):
        dummy = count % 2
        val.append(dataset[dummy + 2, i])
        train.append(dataset[(1 - dummy) + 2, i])
        count = count + 1
    return np.array(train), np.array(val)


def sent2matrix(sentence, dictionary):
    words = sentence.split()
    m = np.int32(np.zeros((1, len(words))))

    for i in range(len(words)):
        m[0, i] = dictionary[words[i]]
    return m


def matrix2sent(matrix, reverse_dictionary):
    text = ""
    for i in range(matrix.shape[0]):
        text = text + " " + reverse_dictionary[matrix[i]]
    return text


def GetRandomTrajectory(motion):
    length = seq_length
    canvas_size = image_size - digit_size

    y = random.randint(15, 85) / 100  # the starting point of the two numbers
    x = random.randint(15, 85) / 100

    start_y = []
    start_x = []
    start_y.append(y)
    start_x.append(x)

    if motion == 0:
        v_y, v_x = 2., 0
    else:
        v_y, v_x = 0, 2.

    direction = random.choice([1, 0])  # 1 is moving right or down, 0 is moving left or top
    bounce = random.choice([1, 0])
    for i in range(length):
        if direction == 1:
            y += v_y * step_length
            x += v_x * step_length
            if bounce == 0:
                if x >= 1.0:
                    x, v_x = 1.0, 0
                if y >= 1.0:
                    y, v_y = 1.0, 0
            else:
                if x >= 1.0:
                    x, v_x = 1.0, -v_x
                if y >= 1.0:
                    y, v_y = 1.0, -v_y
                if x <= 0:
                    x, v_x = 0, 0
                if y <= 0:
                    y, v_y = 0, 0
        else:
            y -= v_y * step_length
            x -= v_x * step_length
            if bounce == 0:
                if x <= 0:
                    x, v_x = 0, 0
                if y <= 0:
                    y, v_y = 0, 0
            else:
                if x <= 0:
                    x, v_x = 0, -v_x
                if y <= 0:
                    y, v_y = 0, -v_y
                if x >= 1.0:
                    x, v_x = 1.0, 0
                if y >= 1.0:
                    y, v_y = 1.0, 0

        # print x, y
        start_y.append(y)
        start_x.append(x)
        if v_y == 0 and v_x == 0:
            break

    # scale to the size of the canvas.
    start_y = (canvas_size * np.array(start_y)).astype(np.int32)
    start_x = (canvas_size * np.array(start_x)).astype(np.int32)
    # print(start_y.shape)
    return start_y, start_x, direction, bounce


def Overlap(a, b):
    return np.maximum(a, b)


def create_gif(digit_imgs, motion, background):
    # get an array of random numbers for indices
    direction = np.zeros(2)
    bounce = np.zeros(2)
    start_y1, start_x1, direction[0], bounce[0] = GetRandomTrajectory(motion[0])
    start_y2, start_x2, direction[1], bounce[1] = GetRandomTrajectory(motion[1])
    if start_y1.shape[0] < start_y2.shape[0]:
        start_y1 = np.concatenate([start_y1, np.repeat(start_y1[-1], start_y2.shape[0] - start_y1.shape[0])], axis=0)
        start_x1 = np.concatenate([start_x1, np.repeat(start_x1[-1], start_x2.shape[0] - start_x1.shape[0])], axis=0)
    elif start_y1.shape[0] > start_y2.shape[0]:
        start_y2 = np.concatenate([start_y2, np.repeat(start_y2[-1], start_y1.shape[0] - start_y2.shape[0])], axis=0)
        start_x2 = np.concatenate([start_x2, np.repeat(start_x2[-1], start_x1.shape[0] - start_x2.shape[0])], axis=0)
    gifs = np.zeros((start_y1.shape[0], 1, image_size, image_size), dtype=np.float32)
    start_y, start_x = np.concatenate([start_y1.reshape(-1, 1), start_y2.reshape(-1, 1)], axis=1), np.concatenate([start_x1.reshape(-1, 1), start_x2.reshape(-1, 1)], axis=1)
    # print(start_x.shape, start_y.shape)
    for n in range(num_digits):
        digit_image = digit_imgs[n, :, :]
        for i in range(gifs.shape[0]):
            top = start_y[i, n]
            left = start_x[i, n]
            bottom = top + digit_size
            right = left + digit_size
            gifs[i, 0, top:bottom, left:right] = Overlap(gifs[i, 0, top:bottom, left:right], digit_image)
    if_bg = random.choice([0, 1])
    if if_bg == 1:
        top = int((image_size - digit_size) * np.random.rand(1))
        left = int((image_size - digit_size) * np.random.rand(1))
        bottom = top + digit_size
        right = left + digit_size
        box_digits = np.array([start_y[0, :], start_x[0, :], start_y[0, :] + digit_size, start_x[0, :] + digit_size])
        while IOU([top, left, bottom, right], box_digits[:, 0]) or IOU([top, left, bottom, right], box_digits[:, 1]):
            top = int((image_size - digit_size) * np.random.rand(1))
            left = int((image_size - digit_size) * np.random.rand(1))
            bottom = top + digit_size
            right = left + digit_size
        gifs[:, 0, top:bottom, left:right] = Overlap(gifs[:, 0, top:bottom, left:right], background)

    return gifs, direction, bounce

def IOU(box1, box2, threshold = 0.7):
    top_d = max(box1[0], box2[0])
    left_d = max(box1[1], box2[1])
    bottom_d = min(box1[2], box2[2])
    right_d = min(box1[3], box2[3])
    iterbox = max(0, right_d - left_d) * max(0, bottom_d - top_d)
    iou = iterbox / float(digit_size ** 2 * 2 - iterbox)
    return True if iou > threshold else False


def create_gifs_for_data(dataset, data, labels, num):
    final_gif_data = []
    outer_index = 0
    inner_digits = dataset % 100
    motion_values = dataset // 100
    while outer_index < num:
        print(outer_index)
        idxs = np.random.randint(data.shape[0], size=num_digits)
        if 10 * labels[idxs[0]] + labels[idxs[1]] in inner_digits:
            n = 10 * labels[idxs[0]] + labels[idxs[1]]
            motion_list = np.where(inner_digits == n)[0]
            random.shuffle(motion_list)
            motion_idx = motion_idxs[motion_values[motion_list[0]]]

            background_idxs = np.random.randint(data.shape[0], size=1)
            while labels[background_idxs] == labels[idxs[0]] or labels[background_idxs] == labels[idxs[1]]:
                background_idxs = np.random.randint(data.shape[0], size=1)
            background = data[background_idxs]

            digit = data[idxs]
            dummy, direction, bounce = create_gif(digit, motion_idx, background)
            direction, bounce = direction.astype(np.int32), bounce.astype(np.int32)

            sentence = 'the digit %d is moving %s and the digit %d is moving %s .' % (
            labels[idxs[0]], motion_strings[motion_idx[0] + 2 * direction[0] + 4 * bounce[0]],
            labels[idxs[1]], motion_strings[motion_idx[1] + 2 * direction[1] + 4 * bounce[1]])
            instance = {'video': dummy, 'caption': sentence}
            final_gif_data.append(instance)
        else:
            outer_index -= 1
        outer_index += 1

    return final_gif_data


if __name__ == "__main__":
    import tensorflow as tf

    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_data = train_x
    train_labels = train_y
    val_data = test_x
    val_labels = test_y

    data = np.concatenate((train_data, val_data), axis=0)
    labels = np.concatenate((train_labels, val_labels), axis=0)

    train, val = create_dataset()
    # print train, val
    data_train = create_gifs_for_data(train, data, labels, 24000)
    data_val = create_gifs_for_data(val, data, labels, 6000)

    if not os.path.exists('./data/moving_mnist'):
        os.makedirs('./data/moving_mnist')

    map_size = 1099511627776 * 2 if platform.system() == "Linux" else 1280000
    db = lmdb.open(
        './data/moving_mnist/mnist_double_modified_20f_30k_train.lmdb', map_size=map_size, subdir=False, meminit=False, map_async=True
    )
    INSTANCE_COUNTER: int = 0
    txn = db.begin(write=True)
    for instance in data_train:
        instance = (instance["video"], instance["caption"])
        txn.put(
            f"{INSTANCE_COUNTER}".encode("ascii"),
            pickle.dumps(instance, protocol=-1)
        )
        INSTANCE_COUNTER += 1
    txn.commit()
    db.sync()
    db.close()

    db2 = lmdb.open(
        './data/moving_mnist/mnist_double_modified_20f_30k_test.lmdb', map_size=map_size, subdir=False,
        meminit=False, map_async=True
    )
    INSTANCE_COUNTER: int = 0
    txn = db2.begin(write=True)
    for instance in data_val:
        instance = (instance["video"], instance["caption"])
        txn.put(
            f"{INSTANCE_COUNTER}".encode("ascii"),
            pickle.dumps(instance, protocol=-1)
        )
        INSTANCE_COUNTER += 1
    txn.commit()
    db2.sync()
    db2.close()

    print('Finished!')

