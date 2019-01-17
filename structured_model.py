
import numpy as np
from random import shuffle
import string
import seaborn as sns
import matplotlib.pyplot as plt

w = np.random.uniform(low=-0.08, high=0.08, size=(26 * 128 + 27 * 27))
w_sum = w.copy()
n_update = 0

letters = string.ascii_lowercase + '$'
letter_to_ix = {letter: i for i, letter in enumerate(letters)}


def phi(xi, prev, curr):
    wi = np.zeros(26 * 128)

    wi[curr * 128: (curr + 1) * 128] = xi

    bi = np.zeros(27 * 27)

    bi[prev * 27 + curr] = 1

    wi = np.concatenate((wi, bi))

    return wi


def word_predict(word, weights):
    d_s = np.zeros((len(word), 27))
    d_pi = np.zeros((len(word), 27))

    prev_char = letter_to_ix['$']
    x = word[0][0]

    for i in range(len(letters) - 1):
        curr_char = letter_to_ix[letters[i]]
        p = phi(x, prev_char, curr_char)
        s = np.dot(weights, p)
        d_s[0][i] = s
        d_pi[0][i] = 0

    for i in range(1, len(word)):
        x = word[i][0]

        for j in range(len(letters) - 1):
            curr_char = letter_to_ix[letters[j]]
            d_best = -1
            i_best = -1

            for k in range(len(letters) - 1):
                y_t = letter_to_ix[letters[k]]
                tmp_d = np.dot(weights, phi(x, y_t, curr_char)) + d_s[i - 1][y_t]

                if tmp_d > d_best:
                    d_best = tmp_d
                    i_best = y_t

            d_s[i][j] = d_best
            d_pi[i][j] = i_best

    y_hat = np.zeros(len(word))
    d_best = -1
    for i in range(len(letters) - 1):
        if d_best < d_s[len(word) - 1][i]:
            y_hat[len(word) - 1] = i
            d_best = d_s[len(word) - 1][i]

    for i in range(len(word) - 2, -1, -1):
        y_hat[i] = d_pi[i + 1][int(y_hat[i + 1])]

    return y_hat


def main():
    global w, w_sum, n_update

    with open('data/letters.train.data') as r_file:
        train_content = r_file.readlines()

    with open('data/letters.test.data') as r_file:
        test_content = r_file.readlines()

    train_content = [line.split() for line in train_content]
    test_content = [line.split() for line in test_content]

    train_data = []
    word = []
    prev_word_idx = None
    for line in train_content:
        word_idx = int(line[3])
        if prev_word_idx is None:
            prev_word_idx = word_idx

        if word_idx == prev_word_idx:
            word.append((np.array([int(b) for b in line[6:]]), letter_to_ix[line[1]]))
        else:
            train_data.append(word.copy())
            word = []
            prev_word_idx = word_idx
    train_data.append(word.copy())

    test_data = []
    word = []
    prev_word_idx = None
    for line in test_content:
        word_idx = int(line[3])
        if prev_word_idx is None:
            prev_word_idx = word_idx

        if word_idx == prev_word_idx:
            word.append((np.array([int(b) for b in line[6:]]), letter_to_ix[line[1]]))
        else:
            word.append((np.array([int(b) for b in line[6:]]), letter_to_ix[line[1]]))
            test_data.append(word.copy())
            word = []
            prev_word_idx = word_idx
    test_data.append(word.copy())

    for e in range(3):
        shuffle(train_data)

        for word_idx, word in enumerate(train_data):
            y_hat_vec = word_predict(word, w)

            for i in range(len(word)):
                x = word[i][0]
                y = word[i][1]
                y_hat = y_hat_vec[i]

                if i == 0:
                    prev_y = letter_to_ix['$']
                    prev_y_hat = letter_to_ix['$']
                else:
                    prev_y = word[i - 1][1]
                    prev_y_hat = y_hat_vec[i - 1]

                n_update += 1
                w = w + phi(x, int(prev_y), int(y)) - phi(x, int(prev_y_hat), int(y_hat))
                w_sum = np.add(w_sum, w)

            if word_idx % (int(len(train_data) / 100)) == 0:
                print(str(e) + '. ' + str(100. * word_idx / len(train_data)) + '%')

    w_sum /= n_update

    acc = 0
    n_letters = 0
    pred_list = []
    for word in test_data:
        y_hat_vec = word_predict(word, w_sum)

        for i in range(len(word)):
            n_letters += 1
            pred_list.append(int(y_hat_vec[i]))
            if int(word[i][1]) == int(y_hat_vec[i]):
                acc += 1

    pred_list = [chr(p + ord('a')) for p in pred_list]
    pred_str = '\n'.join(pred_list)
    with open('structured.pred', 'w') as w_file:
        w_file.write(pred_str)

    print('Test set accuracy: {}%'.format(100. * acc / n_letters))

    bigram_values = np.array(w_sum[-27 * 27:])
    bigram_values = (bigram_values - min(bigram_values)) / (max(bigram_values) - min(bigram_values))
    bigram_values = bigram_values.reshape((27, 27))

    plt.figure()
    ax = sns.heatmap(bigram_values, vmin=0, vmax=1, cmap="YlGnBu")
    plt.xticks(np.arange(len(letters)) + 0.5, list(letters), va="center")
    plt.yticks(np.arange(len(letters)) + 0.5, list(letters), va="center")
    plt.title('bi-gram heatmap')
    plt.show()
    plt.savefig('bigarm_heat-map')


if __name__ == '__main__':
    main()
