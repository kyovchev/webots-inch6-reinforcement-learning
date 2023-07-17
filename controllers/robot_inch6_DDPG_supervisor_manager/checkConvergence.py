import matplotlib.pyplot as plt
import numpy as np

file = open("./exports/Episode-score.txt", "r")
lines = file.read().splitlines()
scores = list(map(float, lines))
episode = list(range(1, 1 + len(scores)))
plt.figure()
plt.title("Episode scores over episode")
plt.plot(episode, scores, label='Raw data')
simple_moving_average = np.convolve(scores, np.ones(500), 'valid') / 500
plt.plot(simple_moving_average, label='SMA500')
plt.xlabel("episode")
plt.ylabel("episode score")
plt.legend()
plt.savefig('./exports/trend.png')
print("Last SMA500:", np.mean(scores[-500:]))
