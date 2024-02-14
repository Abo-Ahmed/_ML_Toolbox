import matplotlib.pyplot as plt
from collections import Counter

def create_frequency_table(data):
    # Step 1: Calculate the frequency of each class
    class_counts = Counter(data)

    # Step 2: Print the frequency table
    print("Frequency Table:")
    print("----------------")
    print("Class\t\tFrequency")
    print("----------------")

    for class_value, frequency in class_counts.items():
        print(f"{class_value}\t\t{frequency}")

    print("----------------")

    # Step 3: Draw a bar chart
    classes = list(class_counts.keys())
    frequencies = list(class_counts.values())

    plt.bar(classes, frequencies)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.show()

# Example usage
data = ['O', 'O', 'A', 'B', 'A', 'O', 'A', 'A', 'A', 'O', 'B', 'O', 'B', 'O', 'O', 'A', 'O', 'O', 'A', 'A', 'A', 'A', 'AB', 'A', 'B', 'A', 'A', 'O', 'O', 'A', 'O', 'O', 'A', 'A', 'A', 'O', 'A', 'O', 'O', 'AB']
print("Data type: Nominal")
create_frequency_table(data)