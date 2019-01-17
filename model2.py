
import numpy as np
from random import shuffle


def phi(xi, yi):
    wi = np.zeros(27 * 128)

    wi[yi * 128: (yi + 1) * 128] = xi

    return wi


def main():
    with open('data/letters.train.data') as r_file:
        train_content = r_file.readlines()

    with open('data/letters.test.data') as r_file:
        test_content = r_file.readlines()

    train_content = [line.split() for line in train_content]
    test_content = [line.split() for line in test_content]

    train_data = [(np.array([int(b) for b in line[6:]]), ord(line[1]) - ord('a')) for line in train_content]
    test_data = [(np.array([int(b) for b in line[6:]]), ord(line[1]) - ord('a')) for line in test_content]

    w = np.random.uniform(low=-0.08, high=0.08, size=(27 * 128))
    w_sum = w.copy()
    n_update = 0

    for e in range(3):
        shuffle(train_data)

        for x, y in train_data:
            max = np.dot(w, phi(x, 0))
            arg_max = 0
            for y_hat in range(1, 27):
                score = np.dot(w, phi(x, y_hat))
                if score > max:
                    arg_max = y_hat
                    max = score

            n_update += 1
            w = w + phi(x, y) - phi(x, arg_max)
            w_sum = np.add(w_sum, w)

    w_sum /= n_update

    acc = 0
    for x, y in test_data:
        max = np.dot(w, phi(x, 0))
        arg_max = 0
        for y_hat in range(1, 27):
            score = np.dot(w_sum, phi(x, y_hat))
            if score > max:
                arg_max = y_hat
                max = score

        if arg_max == y:
            acc += 1

    print('Test set accuracy: {}%'.format(100. * acc / len(test_data)))


if __name__ == '__main__':
    main()
