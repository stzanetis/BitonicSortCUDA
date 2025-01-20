import subprocess
import matplotlib.pyplot as plt

# Versions of the CUDA code
versions = ["V0", "V1", "V2", "Qsort"]
# Range of N values
N_values = range(20, 30)
# Dictionary to store compile times for each version
exec_times = {version: [] for version in versions}

# Function to compile the CUDA code
def compile(version):
    subprocess.run(["cd", "C:\\Users\\Savvas\\Documents\\GitHub Repos\\BitonicSortCUDA\\Testing"], shell=True, check=True)
    subprocess.run(["make", "-C", f"../{version}"], check=True)

# Function to run the CUDA executable
def run(version, N):
    result = subprocess.run([f"..\\{version}\\{version}.exe", str(N)], check=True, shell=True, capture_output=True, text=True)
    output = result.stdout
    for line in output.splitlines():
        if "took" in line:
            exec_time = float(line.split("took")[1].strip().split()[0])
            return exec_time
    raise ValueError("Execution time not found in the output")

# Compile each version once
for version in versions:
    compile(version)

# Test each version for each N value
for version in versions:
    for N in N_values:
        exec_time = run(version, N)
        print(f"{version} Execution Time for N = {N}: {exec_time} seconds")
        exec_times[version].append(exec_time)

# Plot the results
plt.figure(figsize=(15, 9))
for version in versions:
    plt.plot(N_values, exec_times[version], label=f"{version} Execution Time")

plt.xlabel("N (log(2) elements)")
plt.ylabel("Compile Time (seconds)")
plt.title("Execution Time Using GeForce RTX 3060")
plt.legend()
plt.grid(True)
plt.show()