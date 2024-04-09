import tensorflow as tf
import math

class FFT:
    def __init__(self, size: int) -> None:
        """
        Initialize an FFT (Fast Fourier Transform) instance with a specified size.
        Use the `compute` method compute the FFT using the Cooley-Tukey algorithm.

        Args:
            size (int): The size of the FFT, which must be a power of 2.
        """
        self.size = size
        self.stages = int(math.log2(size))

        self.bit_rev_indices = tf.constant(self.bit_rev_list(size))

        # Create test twiddle factors for each stage
        self.twiddles = [tf.ones([2, 2**k]) for k in range(self.stages)]

    def bit_rev_list(self, list_size: int) -> list[int]:
        """
        Generates a bit-reversed list with the specified size.

        Args:
            list_size (int): The size of the output list, expected to be a power of two.

        Returns:
            list[int]: A bit-reversed list of the specified size.
        """
        # Calculate the number of iterations needed for the desired list size
        steps = int(math.log2(list_size)) - 1

        # Initialize the list with the first two elements as the base case.
        array = [0, 1]

        for _ in range(steps):
            # Double the current elements (equivalent to a left bit shift),
            # then append these elements plus 1 to simulate bit reversal
            array = [element * 2 for element in array] + \
                    [element * 2 + 1 for element in array]

        # Return the constructed bit-reversed list
        return array

    def twiddles():
        pass

    def butterfly(self, tensor: tf.Tensor, stage: int) -> tf.Tensor:
        """
        Perform a butterfly operation on a tensor for Fast Fourier Transform (FFT).

        This method applies the butterfly operation, crucial for FFT algorithms,
        by splitting the input tensor into 'even' and 'odd' components and then
        recombining them with applied twiddle factors for the given FFT stage.

        Args:
            tensor (tf.Tensor): A 3D TensorFlow tensor with the shape (2, A, B).
            stage (int): The current FFT stage, used to select the appropriate twiddle factors.

        Returns:
            tf.Tensor: A TensorFlow tensor of the same shape as `tensor`, with the butterfly operation applied.
        
        Description of the input tensor's shape:
        - First dimension of size 2 represents real and imaginary parts of complex numbers.
        - A denotes the number of parallel butterfly operations.
        - B is the size of each butterfly operation.
        - A * B should equal `self.size`, the total number of elements in the input tensor.
        """
        # Visualization of the butterfly operation: https://tinyurl.com/radix2-butterfly, where:
        #   a represents the even part of the input
        #   b represents the odd part of the input
        #   Wn represents the twiddle factors

        # Split the input tensor into its even (a) and odd (b) components
        evens, odds = tf.split(tensor, 2, axis=2)

        # Apply twiddle factors (Wn) to odds (b) using complex multiplication:
        #   Calculate real component: (a * c) − (b * d)
        #   Calculate imaginary component: (a * d) + (b * c)
        #   Here, odds corresponds to (a, b) and self.twiddles[stage] corresponds to (c, d)
        # Additional information:
        #   The [0] index refers to the real component, and the [1] index refers to the imaginary component
        #   Twiddle factors are complex numbers that rotate the odds in the complex plane
        #   Broadcasting allows this operation to efficiently apply the twiddle factors to multiple tensors
        odds = tf.stack([ 
            odds[0] * self.twiddles[stage][0] - odds[1] * self.twiddles[stage][1], # Real part
            odds[0] * self.twiddles[stage][1] + odds[1] * self.twiddles[stage][0]  # Imag part
        ])

        # The final butterfly operation combines the evens with the twiddled odds
        return tf.concat([evens + odds, evens - odds], axis=2)

    @tf.function
    def compute(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Compute the Fast Fourier Transform (FFT) of the input tensor using the Cooley-Tukey algorithm.

        Args:
            tensor (tf.Tensor): Input tensor with shape (2, self.size),
            where the first dimension of size 2 represents the real and imaginary parts of complex numbers.

        Returns:
            tf.Tensor: Transformed tensor representing the FFT result with shape (2, self.size).
        """
        # Reorder the input using the bit-reversed indices
        # Note: This operation is not supported by the Edge TPU (TODO)
        tensor = tf.gather(tensor, self.bit_rev_indices, axis=1)

        # Apply Cooley-Tukey FFT algorithm using butterfly operations
        for stage in range(self.stages):
            # Calculate butterfly size and count based on the current stage
            butterfly_size = 2 ** (stage + 1)
            butterfly_count = self.size // butterfly_size

            # Reshape the tensor to prepare for butterfly operations
            tensor = tf.reshape(tensor, [2, butterfly_count, butterfly_size])
            
            # Perform butterfly operations for the current stage
            tensor = self.butterfly(tensor, stage)

        # Reshape the tensor back to the original shape after all stages are processed
        return tf.reshape(tensor, [2, self.size])
