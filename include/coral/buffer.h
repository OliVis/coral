#ifndef BUFFER_H
#define BUFFER_H

#include <cstdint>
#include <mutex>
#include <condition_variable>

/**
 * @brief Thread-safe circular buffer implementation for storing byte arrays.
 *
 * This class provides a thread-safe circular buffer that can store fixed-size byte arrays.
 * When the buffer is full and a new element is added, the oldest element is dropped.
 * Threads attempting to retrieve elements from an empty buffer will wait until an element becomes available.
 */
class CircularBuffer {
public:
    /**
     * @brief Constructs a CircularBuffer object.
     * @param element_size Size of each element in bytes.
     * @param capacity Maximum number of elements that the buffer can hold.
     */
    CircularBuffer(const size_t element_size, const int capacity);

    /**
     * @brief Destructs the CircularBuffer object, releasing associated resources.
     */
    ~CircularBuffer();

    /**
     * @brief Puts an element into the buffer.
     * @param source_buffer Pointer to the source data buffer containing the element to be stored.
     *                      The buffer must have a size equal to `element_size`.
     */
    void put(const uint8_t* source_buffer);

    /**
     * @brief Gets an element from the buffer.
     * @param destination_buffer Pointer to the destination buffer to store the retrieved data.
     *                           The buffer must have a size equal to `element_size`.
     */
    void get(uint8_t* destination_buffer);

private:
    uint8_t* buffer;           // Pointer to the underlying buffer.
    const size_t element_size; // Size of each element in bytes
    const int capacity;        // Maximum capacity of the circular buffer

    int head_index; // Index where the next element will be inserted
    int tail_index; // Index of the oldest element in the buffer
    int count;      // Current number of elements in the buffer

    std::mutex mutex;                  // Mutex for thread safety
    std::condition_variable condition; // Condition variable for signaling
};

#endif // BUFFER_H
