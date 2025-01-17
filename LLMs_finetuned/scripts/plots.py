import matplotlib.pyplot as plt
import numpy as np

# Data for the models
models = {
    "Rollama2": {
        "precision": [0.80, 0.82],
        "recall": [0.82, 0.80],
        "f1": [0.81, 0.81],
        "accuracy": 0.81,
        "time": 3704.6796,
    },
    "Romistral": {
        "precision": [0.81, 0.83],
        "recall": [0.83, 0.80],
        "f1": [0.82, 0.81],
        "accuracy": 0.82,
        "time": 3682.1048,
    },
    "Rogemma2": {
        "precision": [0.79, 0.86],
        "recall": [0.87, 0.77],
        "f1": [0.83, 0.81],
        "accuracy": 0.82,
        "time": 6121.1224,
    },
    "Rogemma2 Base": {
        "precision": [0.71, 0.89],
        "recall": [0.93, 0.63],
        "f1": [0.81, 0.74],
        "accuracy": 0.78,
        "time": None,
    },
}

# Labels for classes
classes = ["Pozitiv", "Negativ"]

def plot_individual_metrics(model_name, metrics):
    """Plots precision, recall, and f1-score for a single model."""
    x = np.arange(len(classes))
    width = 0.2

    fig, ax = plt.subplots()
    ax.bar(x - width, metrics["precision"], width, label="Precision")
    ax.bar(x, metrics["recall"], width, label="Recall")
    ax.bar(x + width, metrics["f1"], width, label="F1-Score")

    ax.set_xlabel("Classes")
    ax.set_ylabel("Scores")
    ax.set_title(f"Metrics for {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_comparison_accuracy_and_time(models):
    """Plots accuracy and training time comparison for all models."""
    names = list(models.keys())
    accuracies = [models[model]["accuracy"] for model in names]
    times = [models[model]["time"] if models[model]["time"] else 0 for model in names]

    x = np.arange(len(names))
    width = 0.4

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.bar(x - width / 2, accuracies, width, label="Accuracy", color="b")
    ax2.bar(x + width / 2, times, width, label="Training Time", color="g")

    ax1.set_xlabel("Models")
    ax1.set_ylabel("Accuracy", color="b")
    ax2.set_ylabel("Training Time (s)", color="g")
    ax1.set_title("Accuracy and Training Time Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_model_comparison_detailed(models, metric):
    """Compares a specific metric (precision, recall, or f1) across all models, including class-specific values."""
    names = list(models.keys())
    positive_values = [models[model][metric][0] for model in names]
    negative_values = [models[model][metric][1] for model in names]
    general_values = [np.mean(models[model][metric]) for model in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width, positive_values, width, label=f"Positive {metric.capitalize()}", color="c")
    ax.bar(x, negative_values, width, label=f"Negative {metric.capitalize()}", color="m")
    ax.bar(x + width, general_values, width, label=f"Average {metric.capitalize()}", color="y")

    ax.set_xlabel("Models")
    ax.set_ylabel(f"{metric.capitalize()} Score")
    ax.set_title(f"Detailed Comparison of {metric.capitalize()} Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_model_comparison_accuracy(models):
    """Compares accuracy across all models, including overall and class-specific values."""
    names = list(models.keys())
    positive_recall = [models[model]["recall"][0] for model in names]
    negative_recall = [models[model]["recall"][1] for model in names]
    accuracies = [models[model]["accuracy"] for model in names]

    x = np.arange(len(names))
    width = 0.3

    fig, ax = plt.subplots()
    ax.bar(x - width, positive_recall, width, label="Positive Recall", color="c")
    ax.bar(x, negative_recall, width, label="Negative Recall", color="m")
    ax.bar(x + width, accuracies, width, label="Overall Accuracy", color="y")

    ax.set_xlabel("Models")
    ax.set_ylabel("Scores")
    ax.set_title("Accuracy and Recall Comparison Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# Plotting individual metrics
for model_name, metrics in models.items():
    plot_individual_metrics(model_name, metrics)

# Plot accuracy and training time comparison
plot_comparison_accuracy_and_time(models)

# Plot comparisons of precision, recall, f1-score, and accuracy with detailed metrics
for metric in ["precision", "recall", "f1"]:
    plot_model_comparison_detailed(models, metric)

plot_model_comparison_accuracy(models)
