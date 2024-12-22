import re
import csv

# Read the log data from the file "results.txt"
with open('martin_results.txt', 'r') as file:
    log_data = file.read()

# Define the regex pattern to extract the relevant data
pattern = re.compile(
    r'Fitness Function: ([\w\s]+), Constraint: ([\w\s\d]+), Constraints Used: [\w]+, '
    r'Constraint Handling Method: ([\w\s]+), Particle Inertia: ([\d.]+), '
    r'Personal Confidence: ([\d.]+), Swarm Confidence: ([\d.]+), '
    r'Population Size: (\d+), Particle Speed Limit: (\d+), '
    r'Final Fitness: ([\d.e-]+), Optimum Found After: (\d+), Iterations: (\d+)'
)

# Find all matches using the regex pattern
matches = pattern.findall(log_data)

# Prepare the header for the CSV file
csv_header = [
    'Fitness Function', 'Constraint', 'Constraint Handling Method', 
    'Particle Inertia', 'Personal Confidence', 'Swarm Confidence', 
    'Population Size', 'Particle Speed Limit', 'Final Fitness', 
    'Optimum Found After', 'Iterations'
]

# Open a CSV file to write the results
with open('martin_results.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write the header to the CSV file
    csv_writer.writerow(csv_header)
    
    # Write the filtered data for each match
    for match in matches:
        row = [
            match[0].strip(),          # Fitness Function
            match[1].strip(),          # Constraint
            match[2].strip(),          # Constraint Handling Method
            float(match[3]),           # Particle Inertia
            float(match[4]),           # Personal Confidence
            float(match[5]),           # Swarm Confidence
            int(match[6]),             # Population Size
            int(match[7]),             # Particle Speed Limit
            float(match[8]),           # Final Fitness
            int(match[9]),             # Optimum Found After
            int(match[10])             # Iterations
        ]
        csv_writer.writerow(row)

print("Results saved to martin_results.csv")
