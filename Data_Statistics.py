import csv
import numpy as np

sampleSize = 97
t = 1.69 # 95% confidence interval

# Load the data - SGD
SGD_data = np.transpose(np.loadtxt("SGD_Data.csv", delimiter=','))

# Training data
SGD_training_mean = np.mean(SGD_data[0])
SGD_training_std = np.std(SGD_data[0])
SGD_training_conf = f"{SGD_training_mean} +/- {t*(SGD_training_std/np.sqrt(sampleSize))}"

# Test data
SGD_test_mean = np.mean(SGD_data[1])
SGD_test_std = np.std(SGD_data[1])
SGD_test_conf = f"{SGD_test_mean} +/- {t*(SGD_test_std/np.sqrt(sampleSize))}"

# Load the data - ADAM
ADAM_data = np.transpose(np.loadtxt("ADAM_Data.csv", delimiter=','))

# Training data
ADAM_training_mean = np.mean(ADAM_data[0])
ADAM_training_std = np.std(ADAM_data[0])
ADAM_training_conf = f"{ADAM_training_mean} +/- {t*(ADAM_training_std/np.sqrt(sampleSize))}"

# Test data
ADAM_test_mean = np.mean(ADAM_data[1])
ADAM_test_std = np.std(ADAM_data[1])
ADAM_test_conf = f"{ADAM_test_mean} +/- {t*(ADAM_test_std/np.sqrt(sampleSize))}"

print("-- SGD --")
print(f"- Training data:\nMean: {SGD_training_mean}\nStd: {SGD_training_std}\nConf: {SGD_training_conf}")
print(f"- Test data:\nMean: {SGD_test_mean}\nStd: {SGD_test_std}\nConf: {SGD_test_conf}")

print("-- ADAM --")
print(f"- Training data:\nMean: {ADAM_training_mean}\nStd: {ADAM_training_std}\nConf: {ADAM_training_conf}")
print(f"- Test data:\nMean: {ADAM_test_mean}\nStd: {ADAM_test_std}\nConf: {ADAM_test_conf}")