#ifndef RTLSDR_H
#define RTLSDR_H

#include <cstdint>
#include <rtl-sdr.h>

/**
 * @brief A basic wrapper class for controlling an RTL-SDR device.
 */
class RtlSdr {
public:
    /**
     * @brief Constructs an RtlSdr object.
     * @param device_index The index of the RTL-SDR device to use (default is 0).
     */
    RtlSdr(const uint32_t device_index = 0);

    /**
     * @brief Destructs the RtlSdr object, releasing associated resources.
     */
    ~RtlSdr();

    /**
     * @brief Sets the sample rate of the device.
     * @param rate The sample rate in Hz.
     * @return 0 on success, non-zero on error.
     */
    int set_sample_rate(const uint32_t rate);

    /**
     * @brief Sets the center frequency for the device.
     * @param freq The center frequency in Hz.
     * @return 0 on success, non-zero on error.
     */
    int set_center_freq(const uint32_t freq);

    /**
     * @brief Sets the gain for the device.
     *        The gain value will be rounded to the nearest value supported by the device.
     * @param gain Gain in tenths of a dB (e.g., 10 means 1 dB).
     * @return 0 on success, non-zero on error.
     */
    int set_gain(const int gain);
    
    /**
     * @brief Enable automatic gain mode for the device.
     * @return 0 on success, non-zero on error.
     */
    int set_auto_gain();

    /**
     * @brief Enable or disable the internal digital AGC of the RTL2832.
     * @param on true to enable AGC, false to disable AGC.
     * @return 0 on success, non-zero on error.
     */
    int set_agc_mode(const bool on);

private:
    rtlsdr_dev_t* device_ptr; // Pointer to the underlying RTL-SDR device.
};

#endif // RTLSDR_H
