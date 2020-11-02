import pickle
import random

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 16})

def draw_plot_offset(data, offset, edge_color, fill_color):
    pos = np.arange(1, len(data) + 1)+offset
    bp = ax.boxplot(data, positions= pos, widths=0.3, patch_artist=True)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'means', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

    return bp

data_detector = pickle.load(open( "data_detections.pkl", "rb" ))
data_tracker = pickle.load(open( "data_tracker.pkl", "rb" ))

y_data_detector = data_detector[0][1:9]
y_data_tracker = data_tracker[0][1:9]

x_data_detector = data_detector[1][1:9]
x_data_tracker = data_tracker[1][1:9]

size_data_detector = data_detector[2][1:9]
size_data_tracker = data_tracker[2][1:9]

#y_data_detector = [[random.random() for _ in range(50)] for _ in range(9)]
#y_data_tracker = [[random.random() for _ in range(50)] for _ in range(9)]

#x_data_detector = [[random.random() for _ in range(50)] for _ in range(9)]
#x_data_tracker = [[random.random() for _ in range(50)] for _ in range(9)]

#size_data_detector = [[random.random() for _ in range(50)] for _ in range(9)]
#size_data_tracker = [[random.random() for _ in range(50)] for _ in range(9)]


fig = plt.figure()

ax = fig.add_subplot(111)

bp1 = draw_plot_offset(y_data_detector, -0.2, "black", "blue")
bp2 = draw_plot_offset(y_data_tracker, +0.2, "black", "orange")

bins = list(range(1, 9))
ax.set_xticks(bins)
ax.set_xticklabels(map(lambda x: str(x) + "-" + str(x+1), bins))
ax.set_xlabel('Range from camera [m]', fontsize=15)

ax.set_ylim(0, 0.9)
ax.set_ylabel('Absolute error Y [m]', fontsize=15)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Raw detections', 'Tracker'], loc='upper center')

plt.plot()
#plt.show()
# Save the figure
fig.savefig('y_error.png', bbox_inches='tight')

plt.cla()


bp1 = draw_plot_offset(x_data_detector, -0.2, "black", "blue")
bp2 = draw_plot_offset(x_data_tracker, +0.2, "black", "orange")

ax.set_xticks(bins)
ax.set_xticklabels(map(lambda x: str(x) + "-" + str(x+1), bins))
ax.set_xlabel('Range from camera [m]', fontsize=15)

ax.set_ylim(0, 0.9)
ax.set_ylabel('Absolute error X [m]', fontsize=15)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Raw detections', 'tracker'], loc='upper center')
plt.plot()
#plt.show()
# Save the figure
fig.savefig('x_error.png', bbox_inches='tight')

plt.cla()

bp1 = draw_plot_offset(size_data_detector, -0.2, "black", "blue")
bp2 = draw_plot_offset(size_data_tracker, +0.2, "black", "orange")

ax.set_xticks(bins)
ax.set_xticklabels(map(lambda x: str(x) + "-" + str(x+1), bins))
ax.set_xlabel('Range from camera [m]', fontsize=15)

ax.set_ylim(0, 0.9)
ax.set_ylabel('Absolute error diameter [m]', fontsize=15)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Raw detections', 'tracker'], loc='upper center')
plt.plot()
#plt.show()
# Save the figure
fig.savefig('diameter_error.png', bbox_inches='tight')

plt.cla()
