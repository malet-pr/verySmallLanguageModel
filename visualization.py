import matplotlib.pyplot as plt

# Replace these with your actual log values from today
epochs = [9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000]
loss = [2.7385, 2.9745, 2.9364, 2.8951, 2.8464, 2.8548, 2.8244, 2.7833, 2.8417, 2.6898, 2.8527]

plt.figure(figsize=(10, 5))
plt.plot(epochs, loss, marker='o', color='r', label='Training Loss')
plt.title('The Legacy Ghost: Final 1000 Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()