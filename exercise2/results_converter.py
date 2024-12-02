import re
import csv

# Read the log data from the file "results.txt"
with open('results.txt', 'r') as file:
    log_data = file.read()

# Define the regex pattern to extract the relevant data
pattern = re.compile(r'Particle Inertia: ([\d.]+), Personal Confidence: ([\d.]+), Swarm Confidence: ([\d.]+), Population Size: (\d+), Particle Speed Limit: (\d+), Constraint Handling Method: ([\w\s]+).+Final Fitness: ([\d.]+), Optimum Found After: (\d+), Iterations: (\d+)')

# Find all matches using the regex pattern
matches = pattern.findall(log_data)

# Prepare the header for the CSV file
csv_header = [
    'Particle Inertia', 'Personal Confidence', 'Swarm Confidence', 
    'Population Size', 'Particle Speed Limit', 'Constraint Handling Method', 
    'Final Fitness', 'Optimum Found After', 'Iterations'
]

# Open a CSV file to write the results
with open('simulation_results.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write the header to the CSV file
    csv_writer.writerow(csv_header)
    
    # Write the filtered data for each match
    for match in matches:
        row = [
            float(match[0]),  # Particle Inertia
            float(match[1]),  # Personal Confidence
            float(match[2]),  # Swarm Confidence
            int(match[3]),    # Population Size
            int(match[4]),    # Particle Speed Limit
            match[5],         # Constraint Handling Method
            float(match[6]),  # Final Fitness
            int(match[7]),    # Optimum Found After
            int(match[8])     # Iterations
        ]
        csv_writer.writerow(row)

print("Results saved to simulation_results.csv")
