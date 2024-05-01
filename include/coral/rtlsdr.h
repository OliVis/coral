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
     * @brief This alias is used to represent a function pointer type that matches
     *        the signature of a callback function expected by read_async.
     * @param buf Pointer to the data buffer.
     * @param len Length of the data buffer.
     * @param ctx Pointer to context information (optional).
     */
    using rtlsdr_callback_t = void(*)(uint8_t* buf, uint32_t len, void* ctx);

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
     * @param gain The target gain value in dB.
     * @return 0 on success, non-zero on error.
     */
    int set_gain(const float gain);
    
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

    /**
     * @brief Synchronously reads data from the RTL-SDR device into the provided buffer.
     * @param buffer Pointer to the buffer where the read data will be stored.
     * @param num_bytes Number of bytes to read from the device, must be a multiple of 512.
     *                  Ideally should be a multiple of 16384 (URB size).
     * @return 0 on success, non-zero on error.
     */
    int read_sync(uint8_t* buffer, const int num_bytes) const;

    /**
     * @brief Read samples from the RTL-SDR device asynchronously.
     *        This function initiates an asynchronous read operation from the RTL-SDR device
     *        using the specified callback function. The function will block until the read
     *        operation is canceled using `cancel_async()`.
     * @param callback Pointer to the callback function, signature: void callback(uint8_t*, uint32_t, void*).
     * @param context Pointer to optional context data passed to the callback.
     * @param num_bytes Number of bytes to read per buffer, must be a multiple of 512.
     *                  Ideally should be a multiple of 16384 (URB size).
     * @param num_buffers Optional buffer count (default: 15).
     * @return 0 on success, non-zero on error.
     */
    int read_async(rtlsdr_callback_t callback, void* context, const int num_bytes,
                   const int num_buffers = 15) const;

    /**
     * @brief Cancels the ongoing asynchronous read operation initiated by the `read_async` method.
     *        The method will unblock and return once the operation is successfully canceled.
     * @return 0 on success, non-zero on error.
     */
    int cancel_async() const;

private:
    rtlsdr_dev_t* device_ptr; // Pointer to the underlying RTL-SDR device.
};

#endif // RTLSDR_H
