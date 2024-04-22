#include <iostream>
#include <cstring>
#include "coral/buffer.h"

CircularBuffer::CircularBuffer(const size_t element_size, const int capacity)
    : element_size(element_size), capacity(capacity), head_index(0), tail_index(0), count(0) {
    // Allocate memory for the circular buffer
    buffer = new uint8_t[element_size * capacity];
}

CircularBuffer::~CircularBuffer() {
    // Free allocated memory
    delete[] buffer;
}

void CircularBuffer::put(const uint8_t* source_buffer) {
    // Lock the mutex before modifying the buffer
    std::unique_lock<std::mutex> lock(mutex);

    // If buffer is full, drop the oldest sample (overwrite the tail)
    if (count == capacity) {
        std::cerr << "Dropping sample." << std::endl;
        tail_index = (tail_index + 1) % capacity;
        count--;
    }

    // Copy the source data into the buffer at the head
    std::memcpy(buffer + (head_index * element_size), source_buffer, element_size);

    // Update head index and count
    head_index = (head_index + 1) % capacity;
    count++;

    // Unlock mutex before notifying
    lock.unlock();

    // Notify waiting threads that data is available
    condition.notify_one();
}

void CircularBuffer::get(uint8_t* destination_buffer) {
    // Lock the mutex before modifying the buffer
    std::unique_lock<std::mutex> lock(mutex);

    // Wait until there is data available in the buffer (if empty)
    condition.wait(lock, [this] { return count > 0; });

    // Copy the data from the buffer at the tail to the destination buffer
    std::memcpy(destination_buffer, buffer + (tail_index * element_size), element_size);

    // Update tail index and count
    tail_index = (tail_index + 1) % capacity;
    count--;
}
