

# the fast way
import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
y = [5, 6, 7, 8]
plt.plot(x, y)
plt.show()

# the object oriented way
import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
y = [5, 6, 7, 8]
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()

# Create line chart or scatter plot
plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')

# Label x-axis and y-axis
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')

# Add title to plot
plt.title('Plot Title')

# Add legend to plot
plt.legend(['Data1', 'Data2'])

# Set limits of x-axis and y-axis
plt.xlim(0, 10)
plt.ylim(0, 100)

# Set tick marks and labels for x-axis and y-axis
plt.xticks([0, 1, 2, 3, 4], ['A', 'B', 'C', 'D', 'E'])
plt.yticks([0, 50, 100, 150, 200], ['Low', 'Medium', 'High', 'Very High', 'Max'])

# Add grid to plot
plt.grid(True)

# Save plot to file
plt.savefig('plot.png', dpi=300)

# Create figure and subplots
fig, ax = plt.subplots(2, 2)

# Create plot in an object-oriented way
ax.plot(x_values, y_values)
