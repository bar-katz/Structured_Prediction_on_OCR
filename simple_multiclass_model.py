
import numpy as np
from random import shuffle


def main():
    with open('data/letters.train.data') as r_file:
        train_content = r_file.readlines()

    with open('data/letters.test.data') as r_file:
        test_content = r_file.readlines()

    train_content = [line.split() for line in train_content]
    test_content = [line.split() for line in test_content]

    train_data = [(np.array([int(b) for b in line[6:]]), ord(line[1]) - ord('a')) for line in train_content]
    test_data = [(np.array([int(b) for b in line[6:]]), ord(line[1]) - ord('a')) for line in test_content]

    w = np.random.uniform(low=-0.08, high=0.08, size=(26, 128))
    w_sum = w.copy()
    n_update = 0

    for e in range(3):
        shuffle(train_data)

        for x, y in train_data:
            pred = np.dot(w, x)
            y_hat = np.argmax(pred)

            if y_hat != y:
                n_update += 1
                w[y] = np.add(w[y], x)
                w[y_hat] = np.subtract(w[y_hat], x)
                w_sum = np.add(w_sum, w)

    w_sum /= n_update

    pred_list = []
    acc = 0
    for x, y in test_data:
        pred = np.dot(w_sum, x)
        y_hat = np.argmax(pred)

        pred_list.append(int(y_hat))

        if y_hat == y:
            acc += 1

    pred_list = [chr(p + ord('a')) for p in pred_list]
    pred_str = '\n'.join(pred_list)
    with open('multiclass.pred', 'w') as w_file:
        w_file.write(pred_str)

    print('Test set accuracy: {}%'.format(100. * acc / len(test_data)))


if __name__ == '__main__':
    main()
