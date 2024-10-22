import numpy as np
import subprocess
import argparse
import re

class CoralArgs:
    """Encapsulates arguments for the Coral program."""
    def __init__(self, coral_args):
        """Parse and set the arguments as attributes."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-s", "--fft_size", type=int, required=True)
        parser.add_argument("-b", "--batch_size", type=int, required=True)
        parser.add_argument("-o", "--output_file", type=str, required=True)
        parser.add_argument("-i", "--iterations", type=int, required=True)
        parser.add_argument("-d", "--samples_file", type=str)

        args, _ = parser.parse_known_args(coral_args)

        # Set attributes from the parsed arguments
        for key, value in vars(args).items():
            setattr(self, key, value)

def accuracy(coral_args):
    """
    Compare the accuracy of the Coral's FFT implementation with that of NumPy"s FFT.
    """
    # Load processed complex64 from the output data
    output_array = np.fromfile(coral_args.output_file, dtype=np.complex64)

    # Load and transform the raw SDR samples from uint8 to float32
    samples_array = np.fromfile(coral_args.samples_file, dtype=np.uint8)
    samples_array = (samples_array.astype(np.float32) - 127.4) * (1.0 / 128.0)

    # Convert samples into complex64
    samples_array = samples_array.reshape(-1, 2)
    samples_array = samples_array[:, 0] + 1j * samples_array[:, 1]

    # Reshape both arrays for FFT comparison
    output_array = output_array.reshape(coral_args.iterations, coral_args.batch_size, coral_args.fft_size)
    samples_array = samples_array.reshape(coral_args.iterations, coral_args.batch_size, coral_args.fft_size)

    # Perform FFT on the samples array
    samples_array = np.fft.fft(samples_array)

    # Compute the relative difference between the arrays
    difference = np.abs(output_array - samples_array) / np.abs(samples_array) * 100

    # Calculate average, min, and max differences
    avg_diff = np.mean(difference)
    max_diff = np.max(difference)
    min_diff = np.min(difference)

    # Print the accuracy results
    print("\nAccuracy differences:")
    print(f"Avg: {avg_diff:.2f}%")
    print(f"Min: {min_diff:.2f}%")
    print(f"Max: {max_diff:.2f}%")

def performance(output):
    """
    Measures the processing times and counts the number of dropped samples.
    """
    # Count the number of dropped samples using the '!' identifier
    dropped_count = output.count("!")

    # Extract all the input times and output times from the output
    input_times = np.array([float(match) for match in re.findall(r"< (\d+\.\d+)", output)])
    output_times = np.array([float(match) for match in re.findall(r"> (\d+\.\d+)", output)])

    # Calculate the average times
    avg_input = np.mean(input_times)
    avg_output = np.mean(output_times)

    # Display the results
    print("\nPerformance statistics:")
    print(f"Avg input time:  {avg_input:.4f}s")
    print(f"Avg output time: {avg_output:.4f}s")
    print(f"Number samples:  {dropped_count}")

def main():
    # Parse the commandline arguments for the test script
    parser = argparse.ArgumentParser(
        description="Script to test the Coral's FFT accuracy or measure its processing performance."
    )
    parser.add_argument(
        "test_type", choices=["accuracy", "performance"],
        help="Specify the test type: 'accuracy' or 'performance'."
    )
    parser.add_argument(
        "coral_args", nargs=argparse.REMAINDER,
        help="Arguments to pass to the Coral program."
    )
    args = parser.parse_args()

    # Prepare the command to run the Coral program
    command = ["./coral"] + args.coral_args
    command.extend(["-o", "output.dat"])

    # If the test type is "accuracy", also store the SDR samples
    if args.test_type == "accuracy":
        command.extend(["-d", "samples.dat"])

    try:
        # Run the external Coral program and caputure the created data
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)

        # Store the stdout output
        output = result.stdout
    except subprocess.CalledProcessError:
        print("\nError: Coral program execution failed.")
        exit(1)

    # Parse the Coral specific arguments
    coral_args = CoralArgs(command[1:])

    # Perform the specified test
    if args.test_type == "accuracy":
        accuracy(coral_args)
    else:
        performance(output)

if __name__ == "__main__":
    main()
