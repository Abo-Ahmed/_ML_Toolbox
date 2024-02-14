import math
import matplotlib.pyplot as plt

def create_frequency_table(data):
    # Step 1: Find the range
    data_range = max(data) - min(data)
    print("Range: " + str(data_range))
    
    # Step 2: Find the number of classes
    num_classes = int(1 + 3.322 * math.log10(len(data)))
    print("NO. Classes: " + str(num_classes))
    
    # Step 3: Find the class width
    class_width = int( data_range / num_classes)
    print("Class Width: " + str(class_width))
    
    # Step 4: Prepare the frequency table
    frequency_table = {}
    
    for i in range(num_classes):
        lower_bound = min(data) + i * class_width
        upper_bound = lower_bound + class_width
        frequency_table[(lower_bound, upper_bound)] = 0
    
    max_upper_bound = list(frequency_table)[-1][1]
    # Count the frequencies
    for num in data:

        for class_range in frequency_table:
            lower_bound, upper_bound = class_range
            if lower_bound <= num < upper_bound:
                frequency_table[class_range] += 1
            if upper_bound == max_upper_bound and num == upper_bound :
                frequency_table[class_range] += 1
    
    # Step 5: Print the frequency table and calculate the mode
    mode = None
    max_frequency = 0
    max_class = None
    
    print("Frequency Table:")
    print("----------------")
    print("Class\t\tFrequency\tR.Frequency\tMidpoint")
    print("---------------------------------")
    
    for class_range, frequency in frequency_table.items():
        lower_bound, upper_bound = class_range
        midpoint = (lower_bound + upper_bound) / 2
        
        if frequency > max_frequency:
            max_frequency = frequency
            max_class = class_range
            mode = midpoint
        
        print(f"{lower_bound}-{upper_bound}\t\t{frequency}\t\t{frequency / len(data):.2f}\t\t{midpoint:.2f}")
    
    print("---------------------------------")
    print(f"Mode: {mode}")
    print(f"Max Frequency: {max_frequency} for class {max_class}")
    print("---------------------------------")
    
    # Step 6: Draw a histogram
    classes = [f"{lower_bound}-{upper_bound}" for (lower_bound, upper_bound) in frequency_table.keys()]
    frequencies = list(frequency_table.values())
    
    plt.bar(classes, frequencies)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.show()

# Example usage
data = [60 , 63 , 75 , 73  , 86  , 80  , 75  , 65  , 76  , 83  , 90  , 75  , 85  , 90  , 77  , 66  , 65  , 86  , 85 ]
# data = [175 , 184 , 160 , 170 , 173 , 170 , 175 , 165 , 171 , 176 ] 
print("Data type: Quanitative, numerical, continous")
create_frequency_table(data)