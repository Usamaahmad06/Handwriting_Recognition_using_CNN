import matplotlib.pyplot as plt
import numpy as np

# Read predictions from file
predictions_file = "predictions.txt"

# Initialize a dictionary to hold class-wise probabilities
class_probabilities = {}
class_counts = {}

# Read the file and parse the data
with open(predictions_file, "r") as file:
    lines = file.readlines()
    for line in lines:
        # Assume the format is "ClassLabel Probability"
        parts = line.strip().split()
        if len(parts) == 2:
            class_label = parts[0]
            probability = float(parts[1])
            if class_label in class_probabilities:
                class_probabilities[class_label] += probability
                class_counts[class_label] += 1
            else:
                class_probabilities[class_label] = probability
                class_counts[class_label] = 1

# Calculate average probabilities
average_probabilities = {k: class_probabilities[k] / class_counts[k] for k in class_probabilities}

# Sort the classes for plotting
sorted_classes = sorted(average_probabilities.keys())
sorted_probabilities = [average_probabilities[k] for k in sorted_classes]

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.bar(sorted_classes, sorted_probabilities, color='b')

plt.xlabel('Class Labels')
plt.ylabel('Average Prediction Probability')
plt.title('Average Prediction Probability for Each Class')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.show()
