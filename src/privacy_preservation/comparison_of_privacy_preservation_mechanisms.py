import numpy as np
import matplotlib.pyplot as plt

# Given arrays
# 1. (0, 0.4, 0.01)
participant_f1_scores1 = [0.6166666666666668, 0.6166666666666668, 0.5799999999999998, 0.5238888888888888, 0.4344444444444444, 0.2591111111111111, 0.18733333333333335, 0.14823809523809522, 0.10128571428571427, 0.0774126984126984, 0.06851082251082251, 0.05968725718725719, 0.04577730602730602, 0.040013071895424836, 0.017719924812030077, 0.01020228766910689, 0.012142181007398398, 0.011326941657376442, 0.004772663139329806, 0.002526810709419405, 0.005667335644347138, 0.0039824849480021894, 0.002254957006903614, 0.0022329572540918476, 0.002186874304783092]
x_axis_values1 = np.arange(0, 0.25, 0.01)
activity_f1_scores1 = [0.9576719576719576, 0.9150108932461875, 0.9274758792405852, 0.9259477124183008, 0.9003306878306878, 0.8044642857142857, 0.6301182559270795, 0.400635744245584, 0.3257188552188552, 0.24816809116809116, 0.23760724460724458, 0.20668701668701667, 0.16919813902572522, 0.17259852216748767, 0.17189655172413792, 0.16525725232621785, 0.17789363254880497, 0.1684146632422494, 0.16055752038510657, 0.17586070060207987, 0.17789363254880494, 0.17132868089764636, 0.17724046706805324, 0.1613867486281279, 0.1714738794623852]

# Plotting
plt.figure(figsize=(18, 6))
plt.title('Standard 𝜖-differential-Privacy Mechanism')
plt.plot(x_axis_values1, activity_f1_scores1, label='Activity Classification', marker='o')
plt.plot(x_axis_values1, participant_f1_scores1, label='Participant Identification', marker='o')
plt.legend(fontsize=14)

# Adding labels and title
plt.xlabel('Epsilon Value (ϵ)', fontsize=14, fontweight='bold')
plt.ylabel('F1-score', fontsize=14, fontweight='bold')
plt.title('Participant Recognition F1-score with Laplace Noise')
plt.legend(fontsize=16)
plt.xticks(x_axis_values1, labels=np.round(x_axis_values1, 2), fontsize=14, fontweight='bold', rotation=-45)
plt.yticks(np.arange(0, 1.00, 0.05), fontsize=14, fontweight='bold')

plt.grid(True)

# Show the plot
plt.show(block=False)


activity_f1_scores2 = [0.958, 0.958, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.965, 0.965, 0.899, 0.899, 0.899, 0.863, 0.795, 0.795, 0.733, 0.733, 0.696, 0.712, 0.712, 0.712, 0.666, 0.666, 0.666, 0.666, 0.637, 0.599, 0.569]
participant_f1_scores2 = [0.617, 0.617, 0.617, 0.617, 0.617, 0.589, 0.589, 0.511, 0.432, 0.513, 0.513, 0.466, 0.466, 0.43, 0.402, 0.402, 0.361, 0.363, 0.288, 0.204, 0.19, 0.17, 0.118, 0.118, 0.128, 0.137, 0.067, 0.011, 0.011, 0.011, 0.011]
x_axis_values2 = np.arange(0, 0.9, 0.03)

# Plotting
plt.figure(figsize=(18, 6))
plt.title('Proposed Adaptive Feature-based Perturbation (AFP) Mechanism')
plt.plot(x_axis_values2, activity_f1_scores2, label='Activity Classification', marker='o')
plt.plot(x_axis_values2, participant_f1_scores2, label='Participant Identification', marker='o')

# Adding labels and title
plt.xlabel('Targeted Epsilon Value (ϵ)', fontsize=14, fontweight='bold')
plt.ylabel('F1-score', fontsize=14, fontweight='bold')
plt.title('Multi-task Recognition F1-score while adding Laplace Noise to top 1000 most important features for participant recognition')
plt.legend(fontsize=14)
plt.yticks(np.arange(0, 1.05, 0.05), fontsize=14, fontweight='bold')
plt.xticks(x_axis_values2, labels=np.round(x_axis_values2, 2), fontsize=14, fontweight='bold', rotation=-45)

plt.grid(True)

# Show the plot
plt.show()