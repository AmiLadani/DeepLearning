import matplotlib.pyplot as plt

plt.xlabel('Hidden Unit Size')
plt.ylabel('Test Accuracy')

#lstm results
x = [32, 64, 128, 256]
y = [81.87, 83.92, 84.5, 84.29]

a = plt.plot(x, y, label = "LSTM")

#gru results
x = [32, 64, 128, 256]
y = [78.91, 81.21, 81.99, 81.44]

b = plt.plot(x, y, label = "GRU")
plt.legend()
plt.savefig('comparison.png')
