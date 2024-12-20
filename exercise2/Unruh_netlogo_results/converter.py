import re
import csv

# Read the log data from the file "results.txt"
with open('./exercise2/Unruh_netlogo_results/results.txt', 'r') as file:
    log_data = file.read()

# Define the regex pattern to extract the relevant data
pattern = re.compile(r'Fitness Function: (\w+), '
                     r'Constraint: ([\w\s]+), '
                     r'Constraints Used: (\w+), '
                     r'Constraint Handling Method: ([\w\s]+), '
                     r'Particle Inertia: ([\d.]+), '
                     r'Personal Confidence: ([\d.]+), '
                     r'Swarm Confidence: ([\d.]+), '
                     r'Population Size: (\d+), '
                     r'Particle Speed Limit: (\d+), '
                     r'Final Fitness: ([\d.]+), '
                     r'Optimum Found After: (\d+), '
                     r'Iterations: (\d+)')

# Find all matches using the regex pattern
matches = pattern.findall(log_data)

# Prepare the header for the CSV file
csv_header = [
    'Fitness Function', 'Constraints', 'Constraints Used', 'Constraint Handling Method', 
    'Particle Inertia', 'Personal Confidence', 'Swarm Confidence', 
    'Population Size', 'Particle Speed Limit',  
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
            match[0],
            match[1],
            match[2],
            match[3],
            float(match[4]),  # Particle Inertia
            float(match[5]),  # Personal Confidence
            float(match[6]),  # Swarm Confidence
            int(match[7]),    # Population Size
            int(match[8]),    # Particle Speed Limit
            float(match[9]),  # Final Fitness
            int(match[10]),    # Optimum Found After
            int(match[11])     # Iterations
        ]
        csv_writer.writerow(row)

print("Results saved to simulation_results.csv")
