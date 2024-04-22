#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <limits>
#include "coral/rtlsdr.h"

RtlSdr::RtlSdr(const uint32_t device_index) {
    if (rtlsdr_open(&device_ptr, device_index) < 0) {
        throw std::runtime_error("Error: Failed to open the RTL-SDR device.");
    }
    if (rtlsdr_reset_buffer(device_ptr) < 0) {
        throw std::runtime_error("Error: Failed to reset the buffer.");
    }
}

RtlSdr::~RtlSdr() {
    rtlsdr_close(device_ptr);
}

int RtlSdr::set_sample_rate(const uint32_t rate) {
    return rtlsdr_set_sample_rate(device_ptr, rate);
}

int RtlSdr::set_center_freq(const uint32_t freq) {
    return rtlsdr_set_center_freq(device_ptr, freq);
}

int RtlSdr::set_gain(const int gain) {
    int nearest_gain = 0; // Nearest supported gain to the target gain

    // Set manual gain mode
    if (int result = rtlsdr_set_tuner_gain_mode(device_ptr, 1); result < 0) {
        std::cerr << "Error: Failed to enable manual gain mode." << std::endl;
        return result;
    }

    // Get the number of available tuner gains
    const int num_gains = rtlsdr_get_tuner_gains(device_ptr, NULL);

    if (num_gains > 0) {
        // Create a vector to hold the tuner gains
        std::vector<int> gains(num_gains);

        // Fill the vector with supported gain values
        rtlsdr_get_tuner_gains(device_ptr, gains.data());

        int min_diff = std::numeric_limits<int>::max();

        // Find the nearest supported gain to the specified gain
        for (const int& g : gains) {
            int diff = std::abs(g - gain);
            if (diff <= min_diff) {
                nearest_gain = g;
                min_diff = diff;
            }
        }
    }

    // Set the tuner gain to the nearest supported gain
    std::cerr << "Setting tuner gain to " << nearest_gain / 10.0 << "dB." << std::endl;
    return rtlsdr_set_tuner_gain(device_ptr, nearest_gain); 
}

int RtlSdr::set_auto_gain() {
    return rtlsdr_set_tuner_gain_mode(device_ptr, 0);
}

int RtlSdr::set_agc_mode(const bool on) {
    return rtlsdr_set_agc_mode(device_ptr, on);
}

int RtlSdr::read_sync(uint8_t* buffer, const int num_bytes) const {
    if (num_bytes % 512 != 0) {
        throw std::runtime_error("Error: Number of bytes is not a multiple of 512.");
    }

    int bytes_read; // Number of bytes actually read
    int result = rtlsdr_read_sync(device_ptr, buffer, num_bytes, &bytes_read);
    
    // Check if the read operation was successful
    if (result < 0) {
        std::cerr << "Error: Failed to read data from RTL-SDR device." << std::endl;
        return result;
    }

    // Verify that the correct number of bytes were read
    if (bytes_read != num_bytes) {
        throw std::runtime_error("Error: Did not read the requested number of bytes.");
    }

    return result;
}

int RtlSdr::read_async(rtlsdr_callback_t callback, void* context, const int num_bytes,
                       const int num_buffers) const {
    if (num_bytes % 512 != 0) {
        throw std::runtime_error("Error: Number of bytes is not a multiple of 512.");
    }

    // Initiate the asynchronous read operation
    return rtlsdr_read_async(device_ptr, callback, context, num_buffers, num_bytes);
}

int RtlSdr::cancel_async() const {
    return rtlsdr_cancel_async(device_ptr);
}
