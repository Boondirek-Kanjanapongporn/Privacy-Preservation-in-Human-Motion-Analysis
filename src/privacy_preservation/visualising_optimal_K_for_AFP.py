import numpy as np
import matplotlib.pyplot as plt

# K from range 100 to 1000
activity_f1_scores_100 = [0.958, 0.958, 0.958, 0.958, 0.958, 0.958, 0.958, 0.958, 0.958, 0.958, 0.958, 0.929, 0.929, 0.929, 0.929, 0.902, 0.902, 0.864, 0.764, 0.764, 0.764, 0.764, 0.764, 0.74, 0.74, 0.74, 0.74, 0.67, 0.62, 0.584, 0.55]
activity_f1_scores_200 = [0.958, 0.958, 0.958, 0.958, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.965, 0.965, 0.935, 0.935, 0.887, 0.887, 0.857, 0.801, 0.761, 0.777, 0.774, 0.717, 0.717, 0.65, 0.652, 0.656, 0.469, 0.469, 0.442, 0.448]
activity_f1_scores_300 = [0.958, 0.958, 0.958, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.965, 0.965, 0.965, 0.965, 0.868, 0.841, 0.841, 0.841, 0.843, 0.836, 0.725, 0.725, 0.695, 0.672, 0.556, 0.556, 0.514, 0.514, 0.522, 0.522, 0.499]
activity_f1_scores_400 = [0.958, 0.958, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.965, 0.965, 0.965, 0.91, 0.841, 0.841, 0.841, 0.826, 0.83, 0.797, 0.744, 0.744, 0.71, 0.673, 0.673, 0.624, 0.624, 0.599, 0.606, 0.617, 0.531, 0.431]
activity_f1_scores_500 = [0.958, 0.958, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.937, 0.91, 0.882, 0.841, 0.841, 0.841, 0.857, 0.83, 0.806, 0.771, 0.733, 0.71, 0.666, 0.613, 0.576, 0.552, 0.552, 0.552, 0.552, 0.451, 0.464]
activity_f1_scores_600 = [0.958, 0.958, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.965, 0.937, 0.937, 0.882, 0.882, 0.827, 0.827, 0.799, 0.803, 0.735, 0.737, 0.701, 0.63, 0.63, 0.586, 0.586, 0.507, 0.507, 0.477, 0.477, 0.434, 0.434]
activity_f1_scores_700 = [0.958, 0.958, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.965, 0.965, 0.937, 0.868, 0.841, 0.815, 0.815, 0.815, 0.748, 0.764, 0.737, 0.676, 0.558, 0.558, 0.568, 0.535, 0.498, 0.498, 0.456, 0.397, 0.397, 0.397]
activity_f1_scores_800 = [0.958, 0.958, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.965, 0.965, 0.937, 0.896, 0.872, 0.815, 0.803, 0.767, 0.736, 0.726, 0.709, 0.623, 0.623, 0.623, 0.623, 0.527, 0.49, 0.448, 0.389, 0.389, 0.397, 0.464]
activity_f1_scores_900 = [0.958, 0.958, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.965, 0.965, 0.965, 0.91, 0.863, 0.821, 0.802, 0.768, 0.768, 0.737, 0.696, 0.696, 0.683, 0.637, 0.637, 0.6, 0.6, 0.602, 0.515, 0.489, 0.429, 0.432]
activity_f1_scores_1000 = [0.958, 0.958, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.965, 0.965, 0.899, 0.899, 0.899, 0.863, 0.795, 0.795, 0.733, 0.733, 0.696, 0.712, 0.712, 0.712, 0.666, 0.666, 0.666, 0.666, 0.637, 0.599, 0.569]
x_axis_values = np.arange(0, 0.9, 0.03)

participant_f1_scores_100 = [0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.589, 0.597, 0.597, 0.533, 0.489, 0.417, 0.422, 0.389, 0.394, 0.394, 0.394, 0.367, 0.367, 0.333, 0.322, 0.33, 0.33, 0.297, 0.308, 0.274]
participant_f1_scores_200 = [0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.622, 0.608, 0.608, 0.506, 0.506, 0.506, 0.461, 0.456, 0.467, 0.467, 0.467, 0.397, 0.328, 0.33, 0.33, 0.319, 0.28, 0.313, 0.313, 0.313, 0.322, 0.333]
participant_f1_scores_300 = [0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.589, 0.597, 0.597, 0.547, 0.558, 0.524, 0.524, 0.447, 0.45, 0.461, 0.458, 0.447, 0.408, 0.364, 0.364, 0.364, 0.297, 0.294, 0.297, 0.296, 0.196, 0.19, 0.19, 0.19]
participant_f1_scores_400 = [0.617, 0.617, 0.617, 0.617, 0.617, 0.589, 0.589, 0.589, 0.572, 0.547, 0.558, 0.552, 0.508, 0.511, 0.439, 0.444, 0.397, 0.313, 0.313, 0.313, 0.292, 0.292, 0.222, 0.188, 0.163, 0.163, 0.124, 0.124, 0.126, 0.126, 0.113]
participant_f1_scores_500 = [0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.589, 0.597, 0.552, 0.558, 0.569, 0.552, 0.524, 0.474, 0.402, 0.365, 0.362, 0.36, 0.329, 0.329, 0.333, 0.333, 0.297, 0.23, 0.172, 0.148, 0.101, 0.112, 0.089, 0.089, 0.09]
participant_f1_scores_600 = [0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.589, 0.552, 0.558, 0.519, 0.563, 0.558, 0.524, 0.48, 0.388, 0.389, 0.383, 0.354, 0.36, 0.326, 0.326, 0.303, 0.231, 0.213, 0.214, 0.214, 0.184, 0.114, 0.057, 0.056, 0.023]
participant_f1_scores_700 = [0.617, 0.617, 0.617, 0.617, 0.617, 0.617, 0.589, 0.544, 0.508, 0.552, 0.558, 0.521, 0.477, 0.436, 0.352, 0.35, 0.395, 0.373, 0.311, 0.304, 0.26, 0.249, 0.179, 0.163, 0.095, 0.033, 0.01, 0.01, 0.01, 0.009, 0.01]
participant_f1_scores_800 = [0.617, 0.617, 0.617, 0.617, 0.617, 0.589, 0.589, 0.552, 0.43, 0.463, 0.508, 0.466, 0.474, 0.436, 0.352, 0.35, 0.394, 0.389, 0.286, 0.253, 0.226, 0.199, 0.164, 0.094, 0.096, 0.033, 0.032, 0.01, 0.01, 0.011, 0.011]
participant_f1_scores_900 = [0.617, 0.617, 0.617, 0.617, 0.617, 0.589, 0.589, 0.474, 0.432, 0.466, 0.508, 0.466, 0.466, 0.474, 0.402, 0.319, 0.312, 0.312, 0.223, 0.232, 0.229, 0.136, 0.162, 0.127, 0.117, 0.129, 0.1, 0.066, 0.009, 0.009, 0.01]
participant_f1_scores_1000 = [0.617, 0.617, 0.617, 0.617, 0.617, 0.589, 0.589, 0.511, 0.432, 0.513, 0.513, 0.466, 0.466, 0.43, 0.402, 0.402, 0.361, 0.363, 0.288, 0.204, 0.19, 0.17, 0.118, 0.118, 0.128, 0.137, 0.067, 0.011, 0.011, 0.011, 0.011]
x_axis_values = np.arange(0, 0.9, 0.03)

# Plotting
plt.figure(figsize=(18, 6))
plt.title("Finding Optimal K from range 100-1000 (Activity)")
plt.plot(x_axis_values, activity_f1_scores_100, label='k=100', marker='o')
plt.plot(x_axis_values, activity_f1_scores_200, label='k=200', marker='o')
plt.plot(x_axis_values, activity_f1_scores_300, label='k=300', marker='o')
plt.plot(x_axis_values, activity_f1_scores_400, label='k=400', marker='o')
plt.plot(x_axis_values, activity_f1_scores_500, label='k=500', marker='o')
plt.plot(x_axis_values, activity_f1_scores_600, label='k=600', marker='o')
plt.plot(x_axis_values, activity_f1_scores_700, label='k=700', marker='o')
plt.plot(x_axis_values, activity_f1_scores_800, label='k=800', marker='o')
plt.plot(x_axis_values, activity_f1_scores_900, label='k=900', marker='o')
plt.plot(x_axis_values, activity_f1_scores_1000, label='k=1000', marker='o')

# Adding labels and title
plt.xlabel('Targeted Epsilon Value (ϵ)', fontsize=14, fontweight='bold')
plt.ylabel('F1-score', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.yticks(np.arange(0, 1.05, 0.05), fontsize=14, fontweight='bold')
plt.xticks(x_axis_values, labels=np.round(x_axis_values, 2), fontsize=14, fontweight='bold', rotation=-45)

plt.grid(True)

# Show the plot
plt.show(block=False)

# Plotting
plt.figure(figsize=(18, 6))
plt.title("Finding Optimal K from range 100-1000 (Participant)")
plt.plot(x_axis_values, participant_f1_scores_100, label='k=100', marker='o')
plt.plot(x_axis_values, participant_f1_scores_200, label='k=200', marker='o')
plt.plot(x_axis_values, participant_f1_scores_300, label='k=300', marker='o')
plt.plot(x_axis_values, participant_f1_scores_400, label='k=400', marker='o')
plt.plot(x_axis_values, participant_f1_scores_500, label='k=500', marker='o')
plt.plot(x_axis_values, participant_f1_scores_600, label='k=600', marker='o')
plt.plot(x_axis_values, participant_f1_scores_700, label='k=700', marker='o')
plt.plot(x_axis_values, participant_f1_scores_800, label='k=800', marker='o')
plt.plot(x_axis_values, participant_f1_scores_900, label='k=900', marker='o')
plt.plot(x_axis_values, participant_f1_scores_1000, label='k=1000', marker='o')

# Adding labels and title
plt.xlabel('Targeted Epsilon Value (ϵ)', fontsize=14, fontweight='bold')
plt.ylabel('F1-score', fontsize=14, fontweight='bold')
# plt.title('Participant Recognition F1-score with different amount of top k most important features for participant recognition')
plt.legend(fontsize=12)
plt.yticks(np.arange(0, 1.05, 0.05), fontsize=14, fontweight='bold')
plt.xticks(x_axis_values, labels=np.round(x_axis_values, 2), fontsize=14, fontweight='bold', rotation=-45)

plt.grid(True)

# Show the plot
plt.show()


# K from range 1000 to 10000 --------------------------------------------------------------------------------------------


# Given arrays
activity_f1_scores_1000 = [0.958, 0.958, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.965, 0.965, 0.899, 0.899, 0.899, 0.863, 0.795, 0.795, 0.733, 0.733, 0.696, 0.712, 0.712, 0.712, 0.666, 0.666, 0.666, 0.666, 0.637, 0.599, 0.569]
activity_f1_scores_2000 = [0.958, 0.958, 0.957, 0.957, 1.0, 1.0, 1.0, 0.93, 0.93, 0.93, 0.874, 0.818, 0.762, 0.689, 0.624, 0.528, 0.529, 0.47, 0.47, 0.454, 0.395, 0.295, 0.295, 0.295, 0.299, 0.304, 0.304, 0.304, 0.304, 0.284, 0.288]
activity_f1_scores_3000 = [0.958, 0.958, 0.958, 0.958, 1.0, 1.0, 0.911, 0.844, 0.775, 0.689, 0.599, 0.437, 0.438, 0.438, 0.438, 0.35, 0.313, 0.313, 0.313, 0.313, 0.313, 0.313, 0.3, 0.271, 0.255, 0.203, 0.203, 0.203, 0.203, 0.203, 0.118]
activity_f1_scores_4000 = [0.958, 0.958, 0.958, 1.0, 1.0, 0.972, 0.972, 0.874, 0.788, 0.724, 0.596, 0.439, 0.425, 0.414, 0.398, 0.345, 0.269, 0.274, 0.274, 0.254, 0.23, 0.23, 0.178, 0.178, 0.178, 0.178, 0.094, 0.094, 0.094, 0.066, 0.066]
activity_f1_scores_5000 = [0.958, 0.958, 0.958, 0.958, 0.958, 0.972, 0.972, 0.874, 0.844, 0.719, 0.719, 0.635, 0.507, 0.454, 0.402, 0.316, 0.316, 0.299, 0.269, 0.249, 0.249, 0.249, 0.249, 0.249, 0.197, 0.197, 0.197, 0.197, 0.197, 0.113, 0.09]
activity_f1_scores_6000 = [0.958, 0.916, 0.916, 0.916, 0.916, 0.93, 0.93, 0.93, 0.874, 0.844, 0.762, 0.683, 0.637, 0.637, 0.637, 0.522, 0.448, 0.395, 0.395, 0.309, 0.309, 0.309, 0.309, 0.274, 0.257, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26]
activity_f1_scores_7000 = [0.958, 0.867, 0.867, 0.867, 0.957, 0.957, 0.924, 0.924, 0.924, 0.857, 0.789, 0.728, 0.728, 0.652, 0.604, 0.551, 0.466, 0.417, 0.269, 0.269, 0.184, 0.146, 0.068, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
activity_f1_scores_8000 = [0.958, 0.867, 0.844, 0.914, 0.914, 0.836, 0.749, 0.711, 0.634, 0.634, 0.567, 0.454, 0.411, 0.217, 0.217, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.072, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]
activity_f1_scores_9000 = [0.958, 0.867, 0.867, 0.901, 0.822, 0.771, 0.735, 0.752, 0.701, 0.512, 0.418, 0.418, 0.293, 0.293, 0.293, 0.155, 0.155, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]
activity_f1_scores_10000 = [0.958, 0.867, 0.901, 0.901, 0.881, 0.736, 0.711, 0.701, 0.667, 0.587, 0.418, 0.332, 0.293, 0.207, 0.207, 0.207, 0.155, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]
x_axis_values = np.arange(0, 0.90, 0.03)

participant_f1_scores_1000 = [0.617, 0.617, 0.617, 0.617, 0.617, 0.589, 0.589, 0.511, 0.432, 0.513, 0.513, 0.466, 0.466, 0.43, 0.402, 0.402, 0.361, 0.363, 0.288, 0.204, 0.19, 0.17, 0.118, 0.118, 0.128, 0.137, 0.067, 0.011, 0.011, 0.011, 0.011]
participant_f1_scores_2000 = [0.617, 0.617, 0.617, 0.589, 0.552, 0.552, 0.474, 0.469, 0.463, 0.463, 0.419, 0.383, 0.381, 0.336, 0.296, 0.194, 0.159, 0.092, 0.058, 0.058, 0.058, 0.069, 0.08, 0.08, 0.079, 0.079, 0.079, 0.079, 0.079, 0.078, 0.078]
participant_f1_scores_3000 = [0.617, 0.617, 0.589, 0.544, 0.5, 0.402, 0.41, 0.408, 0.31, 0.267, 0.277, 0.241, 0.182, 0.114, 0.092, 0.069, 0.069, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
participant_f1_scores_4000 = [0.617, 0.617, 0.589, 0.5, 0.556, 0.402, 0.427, 0.387, 0.351, 0.235, 0.255, 0.207, 0.126, 0.075, 0.086, 0.036, 0.036, 0.036, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
participant_f1_scores_5000 = [0.617, 0.628, 0.628, 0.506, 0.474, 0.519, 0.374, 0.374, 0.328, 0.327, 0.327, 0.288, 0.254, 0.22, 0.126, 0.068, 0.045, 0.011, 0.01, 0.011, 0.011, 0.011, 0.01, 0.01, 0.01, 0.01, 0.009, 0.009, 0.009, 0.009, 0.009]
participant_f1_scores_6000 = [0.617, 0.617, 0.628, 0.519, 0.441, 0.467, 0.408, 0.348, 0.297, 0.264, 0.23, 0.233, 0.255, 0.249, 0.148, 0.159, 0.081, 0.081, 0.083, 0.081, 0.081, 0.092, 0.092, 0.066, 0.067, 0.067, 0.034, 0.034, 0.044, 0.044, 0.045]
participant_f1_scores_7000 = [0.617, 0.617, 0.617, 0.511, 0.489, 0.467, 0.383, 0.38, 0.324, 0.324, 0.302, 0.225, 0.225, 0.135, 0.118, 0.14, 0.101, 0.064, 0.064, 0.066, 0.066, 0.071, 0.071, 0.071, 0.071, 0.071, 0.071, 0.071, 0.071, 0.066, 0.067]
participant_f1_scores_8000 = [0.617, 0.617, 0.544, 0.508, 0.441, 0.422, 0.33, 0.256, 0.258, 0.21, 0.12, 0.055, 0.043, 0.043, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.011, 0.012, 0.012, 0.014]
participant_f1_scores_9000 = [0.617, 0.617, 0.5, 0.474, 0.438, 0.404, 0.306, 0.287, 0.209, 0.186, 0.229, 0.19, 0.048, 0.07, 0.069, 0.036, 0.036, 0.036, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
participant_f1_scores_10000 = [0.617, 0.589, 0.519, 0.474, 0.406, 0.4, 0.352, 0.272, 0.184, 0.153, 0.152, 0.124, 0.08, 0.091, 0.081, 0.092, 0.092, 0.025, 0.025, 0.036, 0.036, 0.036, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
x_axis_values = np.arange(0, 0.90, 0.03)

# Plotting
plt.figure(figsize=(18, 6))
plt.title("Finding Optimal K from range 1000-10000 (Activity)")
plt.plot(x_axis_values, activity_f1_scores_1000, label='k=1000', marker='o')
plt.plot(x_axis_values, activity_f1_scores_2000, label='k=2000', marker='o')
plt.plot(x_axis_values, activity_f1_scores_3000, label='k=3000', marker='o')
plt.plot(x_axis_values, activity_f1_scores_4000, label='k=4000', marker='o')
plt.plot(x_axis_values, activity_f1_scores_5000, label='k=5000', marker='o')
plt.plot(x_axis_values, activity_f1_scores_6000, label='k=6000', marker='o')
plt.plot(x_axis_values, activity_f1_scores_7000, label='k=7000', marker='o')
plt.plot(x_axis_values, activity_f1_scores_8000, label='k=8000', marker='o')
plt.plot(x_axis_values, activity_f1_scores_9000, label='k=9000', marker='o')
plt.plot(x_axis_values, activity_f1_scores_10000, label='k=10000', marker='o')

# Adding labels and title
plt.xlabel('Targeted Epsilon Value (ϵ)', fontsize=14, fontweight='bold')
plt.ylabel('F1-score', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.yticks(np.arange(0, 1.05, 0.05), fontsize=14, fontweight='bold')
plt.xticks(x_axis_values, labels=np.round(x_axis_values, 2), fontsize=14, fontweight='bold', rotation=-45)

plt.grid(True)

# Show the plot
plt.show(block=False)

# Plotting
plt.figure(figsize=(18, 6))
plt.title("Finding Optimal K from range 1000-10000 (Participant)")
plt.plot(x_axis_values, participant_f1_scores_1000, label='k=1000', marker='o')
plt.plot(x_axis_values, participant_f1_scores_2000, label='k=2000', marker='o')
plt.plot(x_axis_values, participant_f1_scores_3000, label='k=3000', marker='o')
plt.plot(x_axis_values, participant_f1_scores_4000, label='k=4000', marker='o')
plt.plot(x_axis_values, participant_f1_scores_5000, label='k=5000', marker='o')
plt.plot(x_axis_values, participant_f1_scores_6000, label='k=6000', marker='o')
plt.plot(x_axis_values, participant_f1_scores_7000, label='k=7000', marker='o')
plt.plot(x_axis_values, participant_f1_scores_8000, label='k=8000', marker='o')
plt.plot(x_axis_values, participant_f1_scores_9000, label='k=9000', marker='o')
plt.plot(x_axis_values, participant_f1_scores_10000, label='k=10000', marker='o')

# Adding labels and title
plt.xlabel('Targeted Epsilon Value (ϵ)', fontsize=14, fontweight='bold')
plt.ylabel('F1-score', fontsize=14, fontweight='bold')
# plt.title('Participant Recognition F1-score with different amount of top k most important features for participant recognition')
plt.legend(fontsize=12)
plt.yticks(np.arange(0, 1.05, 0.05), fontsize=14, fontweight='bold')
plt.xticks(x_axis_values, labels=np.round(x_axis_values, 2), fontsize=14, fontweight='bold', rotation=-45)

plt.grid(True)

# Show the plot
plt.show()