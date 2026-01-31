import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

def calculate_accuracy(predictions, labels):
    """Calculate classification accuracy"""
    return 100.0 * np.mean(predictions == labels)

def calculate_confusion_matrix(predictions, labels, num_classes):
    """Calculate confusion matrix"""
    return confusion_matrix(labels, predictions, labels=range(num_classes))

def calculate_per_class_accuracy(predictions, labels, num_classes):
    """Calculate accuracy for each class"""
    accuracies = []
    for i in range(num_classes):
        mask = labels == i
        if np.sum(mask) > 0:
            acc = 100.0 * np.mean(predictions[mask] == labels[mask])
            accuracies.append(acc)
        else:
            accuracies.append(0.0)
    return accuracies

def calculate_f1_scores(predictions, labels, num_classes, average='macro'):
    """Calculate F1 scores"""
    return f1_score(labels, predictions, labels=range(num_classes), average=average, zero_division=0)

def calculate_precision_recall(predictions, labels, num_classes, average='macro'):
    """Calculate precision and recall"""
    precision = precision_score(labels, predictions, labels=range(num_classes), 
                               average=average, zero_division=0)
    recall = recall_score(labels, predictions, labels=range(num_classes), 
                         average=average, zero_division=0)
    return precision, recall

def get_classification_report(predictions, labels, class_names):
    """Generate detailed classification report"""
    return classification_report(labels, predictions, target_names=class_names, zero_division=0)

def calculate_snr_stratified_accuracy(predictions, labels, snrs, snr_bins):
    """Calculate accuracy for different SNR ranges"""
    results = {}
    for i in range(len(snr_bins) - 1):
        snr_min, snr_max = snr_bins[i], snr_bins[i + 1]
        mask = (snrs >= snr_min) & (snrs < snr_max)
        if np.sum(mask) > 0:
            acc = calculate_accuracy(predictions[mask], labels[mask])
            results[f"{snr_min}_{snr_max}"] = acc
    return results

def calculate_top_k_accuracy(outputs, labels, k=3):
    """Calculate top-k accuracy"""
    top_k_pred = np.argsort(outputs, axis=1)[:, -k:]
    correct = 0
    for i, label in enumerate(labels):
        if label in top_k_pred[i]:
            correct += 1
    return 100.0 * correct / len(labels)

class MetricsTracker:
    """Track metrics during training"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.labels = []
        self.losses = []
    
    def update(self, predictions, labels, loss=None):
        self.predictions.extend(predictions)
        self.labels.extend(labels)
        if loss is not None:
            self.losses.append(loss)
    
    def get_accuracy(self):
        return calculate_accuracy(np.array(self.predictions), np.array(self.labels))
    
    def get_average_loss(self):
        return np.mean(self.losses) if self.losses else 0.0
    
    def get_confusion_matrix(self, num_classes):
        return calculate_confusion_matrix(np.array(self.predictions), 
                                         np.array(self.labels), num_classes)