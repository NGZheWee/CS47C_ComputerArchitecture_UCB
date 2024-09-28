# Import required libraries
import sys
from collections import deque

# Check for verbose flag in command line arguments
if "-v" in sys.argv:
    verbose = True
    sys.argv.remove("-v")  # Remove the verbose flag from arguments for further processing
else:
    verbose = False

# Check if the correct number of command line arguments are provided
if len(sys.argv) != 3:
    # If not, print usage instructions and exit
    print(f"Usage: {sys.argv[0]} <test case> <thread count> [-v]")
    print("This script finds a solution for a given test case.")
    print("The test case should be a file containing task data.")
    print("The script parses the input file and writes an answer in the specified format.")
    print("You can redirect output to a file using standard command line redirection.")
    print(f"Example: {sys.argv[0]} Test1.txt 2 > answer.out")
    print("The -v (verbose) flag is optional and can be used for debugging.")
    exit()

# Function to parse the test file
def parsetestfile(testfile):
    data = testfile.readlines()[1:]  # Skip the first line and read the rest
    result = []
    for i in data:
        element = [int(j) for j in i.strip().split(" ")]
        result.append({"TaskID": element[0], "runtime": element[1], "prerequisites": element[3:]})
    return result

# Open the test file and parse it
testfile = open(sys.argv[1])
test = parsetestfile(testfile)
testfile.close()  # Close the test file after reading
threadcount = int(sys.argv[2])  # Number of threads/processes

# Function to perform topological sorting on tasks
def topological_sort(tasks):
    in_degree = {i: 0 for i in range(len(tasks))}  # Initialize in-degree of each task
    graph = {i: [] for i in range(len(tasks))}  # Initialize graph

    # Build graph and compute in-degrees
    for task in tasks:
        for prereq in task['prerequisites']:
            graph[prereq].append(task['TaskID'])
            in_degree[task['TaskID']] += 1

    # Queue for nodes with in-degree 0
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    sorted_order = []

    # Perform topological sort
    while queue:
        node = queue.popleft()
        sorted_order.append(node)

        # Reduce in-degree of neighbors and add new 0 in-degree nodes to queue
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return sorted_order

# Function to find a solution for distributing tasks
def findsolution(test, threadcount):
    sorted_tasks = topological_sort(test)  # Sort tasks topologically
    distribution = [[] for _ in range(threadcount)]  # Initialize distribution list for each thread

    # Distribute tasks among threads
    for idx, task in enumerate(sorted_tasks):
        distribution[idx % threadcount].append(task)
    return distribution

# Run the solution and get the result
result = findsolution(test, threadcount)

# Write the result to 'output.txt'
with open('output.txt', 'w') as f:
    val = ""
    for i in result:
        for j in i:
            val += str(j) + ","
        val = val[:-1] + ";"  # Format the output string
    f.write(val[:-1])  # Write the formatted string to the file
