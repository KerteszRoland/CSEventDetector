#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <vector>
#include <stdexcept>

// Windows-specific headers for screenshot
#ifdef _WIN32
#include <windows.h>
#endif

// STB Image library for loading/saving PNG files
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Add these constants at the top of the file, after the includes
#define SCREEN_WIDTH 2560   // 2560p width
#define SCREEN_HEIGHT 1440  // 1440p height

#define DEBUG_MODE false

// Structure to hold event data
struct Event {
    const char* name;
    float last_time;
    const char* message;
    const char* example_img_path;
    unsigned char* image;
    int top_left[2];
    int bottom_right[2];
    unsigned char threshold;
    int diff_threshold;
    float delay;
};

unsigned char* loadImage(const char* filename, int* width, int* height) {
    int channels;
    unsigned char* img = stbi_load(filename, width, height, &channels, 3);
    if (!img) {
        printf("Error loading image %s\n", filename);
        return nullptr;
    }
    return img;
}

void saveImage(const char* filename, unsigned char* image, int width, int height, bool is_grayscale=true) {

 if (!stbi_write_png(filename, 
                     width, 
                     height, 
                     is_grayscale ? 1 : 3,  // 1 channel (grayscale) or 3 channels (RGB)
                     image, 
                     width * (is_grayscale ? 1 : 3))) {  // stride = width for grayscale
     printf("Warning: Failed to save debug image: %s\n", filename);
 } else {
     printf("Saved debug image: %s\n", filename);
 }
}

// Initialize CUDA in main before the main loop
bool initCUDA() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("Error getting device count: %s\n", cudaGetErrorString(error));
        return false;
    }
    
    if (deviceCount == 0) {
        printf("No CUDA devices found\n");
        return false;
    }
    
    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, 0);
    if (error != cudaSuccess) {
        printf("Error getting device properties: %s\n", cudaGetErrorString(error));
        return false;
    }
    
    printf("Using CUDA device: %s\n", deviceProp.name);
    printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    
    error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        printf("Error setting device: %s\n", cudaGetErrorString(error));
        return false;
    }
    
    return true;
}

// CUDA kernel for grayscale conversion and thresholding
__global__ void grayscaleAndThreshold(
    const unsigned char* input,
    unsigned char* output,
    const int width,
    const int height,
    const unsigned char threshold
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    const int x = idx % width;
    const int y = idx / width;
    const int rgb_idx = (y * width + x) * 3;
    
    // RGB to grayscale conversion
    const float gray = 0.299f * input[rgb_idx] + 
                      0.587f * input[rgb_idx + 1] + 
                      0.114f * input[rgb_idx + 2];
    
    // Thresholding
    output[idx] = (threshold > 0 && gray > threshold) ? 255 : 0;
}

// CUDA kernel for image difference calculation
__global__ void calculateDifference(
    const unsigned char* img1,
    const unsigned char* img2,
    const int size,
    int* difference
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    const float diff = abs((int)img1[idx] - (int)img2[idx]);
    atomicAdd(difference, diff);
}

// Function implementation
bool getCroppedGrayThreshImage(
    unsigned char* input,
    unsigned char* output,
    int full_width,
    int* top_left,
    int* bottom_right,
    unsigned char threshold
) {
    if (DEBUG_MODE) {
    printf("Starting getCroppedGrayThreshImage...\n");
    printf("Input pointer: %p\n", (void*)input);
    printf("Output pointer: %p\n", (void*)output);
    printf("Full width: %d\n", full_width);
    printf("Crop region: [%d,%d] to [%d,%d]\n", 
           top_left[0], top_left[1], bottom_right[0], bottom_right[1]);
    }
    
    int crop_width = bottom_right[0] - top_left[0];
    int crop_height = bottom_right[1] - top_left[1];
    if (DEBUG_MODE) {
        printf("Crop dimensions: %dx%d\n", crop_width, crop_height);
    }

    // Validate dimensions
    if (top_left[0] < 0 || top_left[1] < 0 || 
        bottom_right[0] > full_width || 
        crop_width <= 0 || crop_height <= 0) {
        printf("Invalid crop dimensions\n");
        return false;
    }

    int size = crop_width * crop_height;

    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaError_t error;
    
    // Allocate and copy input (RGB data)
    error = cudaMalloc(&d_input, crop_width * crop_height * 3 * sizeof(unsigned char));
    if (error != cudaSuccess) {
        printf("Failed to allocate device memory for input: %s\n", cudaGetErrorString(error));
        return false;
    }

    // Allocate output (grayscale data)
    error = cudaMalloc(&d_output, size * sizeof(unsigned char));
    if (error != cudaSuccess) {
        printf("Failed to allocate device memory for output: %s\n", cudaGetErrorString(error));
        cudaFree(d_input);
        return false;
    }

    // Copy cropped region to device memory
    unsigned char* temp_buffer = (unsigned char*)malloc(crop_width * crop_height * 3);
    if (!temp_buffer) {
        printf("Failed to allocate host memory for temporary buffer\n");
        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    // Copy cropped region to temporary buffer
    for (int y = 0; y < crop_height; y++) {
        for (int x = 0; x < crop_width; x++) {
            int src_idx = ((y + top_left[1]) * full_width + (x + top_left[0])) * 3;
            int dst_idx = (y * crop_width + x) * 3;
            
            temp_buffer[dst_idx] = input[src_idx];        // R
            temp_buffer[dst_idx + 1] = input[src_idx + 1];// G
            temp_buffer[dst_idx + 2] = input[src_idx + 2];// B
        }
    }

    // Copy temporary buffer to device
    error = cudaMemcpy(d_input, temp_buffer, crop_width * crop_height * 3, cudaMemcpyHostToDevice);
    free(temp_buffer);  // Free temporary buffer
    
    if (error != cudaSuccess) {
        printf("Failed to copy input data to device: %s\n", cudaGetErrorString(error));
        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    grayscaleAndThreshold<<<blocks, threadsPerBlock>>>(
        d_input,
        d_output,
        crop_width,
        crop_height,
        threshold
    );

    // Check for kernel errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(error));
        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    // Copy result back to host
    error = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Failed to copy output data from device: %s\n", cudaGetErrorString(error));
        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return true;
}

bool isImgsMatch(
    unsigned char* img,
    unsigned char* example_img,
    int full_width,
    int* top_left,
    int* bottom_right,
    int diff_threshold,
    unsigned char threshold
) {
    if (DEBUG_MODE) {
        printf("\n=== isImgsMatch Debug Info ===\n");
    }
    
    // Validate input parameters
    if (!img) {
        printf("Error: Input image is null\n");
        return false;
    }
    if (!example_img) {
        printf("Error: Example image is null\n");
        return false;
    }
    if (!top_left || !bottom_right) {
        printf("Error: Coordinates are null\n");
        return false;
    }
    if (DEBUG_MODE) {
    printf("Input image pointer: %p\n", (void*)img);
    printf("Example image pointer: %p\n", (void*)example_img);
    printf("Full width: %d\n", full_width);
    printf("Coordinates: [%d,%d] to [%d,%d]\n", 
           top_left[0], top_left[1], bottom_right[0], bottom_right[1]);
    printf("Threshold values - diff: %d, gray: %d\n", diff_threshold, threshold);
    }
    int crop_width = bottom_right[0] - top_left[0];
    int crop_height = bottom_right[1] - top_left[1];
    int size = crop_width * crop_height;
    
    // Validate dimensions
    if (crop_width <= 0 || crop_height <= 0) {
        printf("Error: Invalid crop dimensions: %dx%d\n", crop_width, crop_height);
        return false;
    }
    if (size <= 0) {
        printf("Error: Invalid size: %d\n", size);
        return false;
    }
    
    if (DEBUG_MODE) {
        printf("Crop dimensions: %dx%d (size: %d)\n", crop_width, crop_height, size);
    }
    
    if (DEBUG_MODE) {
        printf("Allocating memory for processed image (%d bytes)...\n", size);
    }
    // Allocate memory for processed image
    unsigned char* processed_img = (unsigned char*)malloc(size);
    if (!processed_img) {
        printf("Error: Failed to allocate memory for processed image\n");
        return false;
    }
    if (DEBUG_MODE) {
        printf("Processed image buffer allocated at: %p\n", (void*)processed_img);
    }

    // Process current screenshot
    if (DEBUG_MODE) {
        printf("Calling getCroppedGrayThreshImage...\n");
    }
    if (!getCroppedGrayThreshImage(
            img,
            processed_img,
            full_width,
            top_left,
            bottom_right,
            threshold)) {
        printf("Error in getCroppedGrayThreshImage\n");
        free(processed_img);
        return false;
    }

    if (DEBUG_MODE) {
        printf("Calculating difference...\n");
    }

    int difference = 0;
    char debug_filename[256];
    snprintf(debug_filename, sizeof(debug_filename), "./images/debug/%s.png", "example_img");
    saveImage(debug_filename, example_img, crop_width, crop_height, true);

    saveImage("./images/debug/processed.png", processed_img, crop_width, crop_height, true);

    for (int i = 0; i < size; i++) {
        float pixel_diff = abs((int)processed_img[i] - (int)example_img[i]);
        difference += pixel_diff;
    }
    difference = (int)(difference / 255.0f);
    
    free(processed_img);
    if (DEBUG_MODE) {
        printf("Difference calculated: %d (threshold: %d)\n", 
        difference, diff_threshold);
    printf("=== End isImgsMatch ===\n");
    }
    printf("%d %d\n", difference, diff_threshold);
    return difference < diff_threshold;
}

#ifdef _WIN32
// Function to capture screenshot on Windows
unsigned char* captureScreen(int* width, int* height) {
    // Use the constants instead of GetSystemMetrics
    *width = SCREEN_WIDTH;   // 2560
    *height = SCREEN_HEIGHT; // 1440
    
    if (DEBUG_MODE) {
        printf("Attempting to capture screen %dx%d\n", *width, *height);
    }

    // Create device context and bitmap
    HDC hScreenDC = GetDC(NULL);
    if (!hScreenDC) {
        printf("Error: GetDC failed with error code %lu\n", GetLastError());
        return nullptr;
    }
    if (DEBUG_MODE) {
        printf("GetDC successful\n");
    }

    HDC hMemoryDC = CreateCompatibleDC(hScreenDC);
    if (!hMemoryDC) {
        printf("Error: CreateCompatibleDC failed with error code %lu\n", GetLastError());
        ReleaseDC(NULL, hScreenDC);
        return nullptr;
    }
    if (DEBUG_MODE) {
        printf("CreateCompatibleDC successful\n");
    }

    HBITMAP hBitmap = CreateCompatibleBitmap(hScreenDC, *width, *height);
    if (!hBitmap) {
        printf("Error: CreateCompatibleBitmap failed with error code %lu\n", GetLastError());
        DeleteDC(hMemoryDC);
        ReleaseDC(NULL, hScreenDC);
        return nullptr;
    }
    if (DEBUG_MODE) {
        printf("CreateCompatibleBitmap successful\n");
    }

    HBITMAP hOldBitmap = (HBITMAP)SelectObject(hMemoryDC, hBitmap);
    if (!hOldBitmap) {
        printf("Error: SelectObject failed with error code %lu\n", GetLastError());
        DeleteObject(hBitmap);
        DeleteDC(hMemoryDC);
        ReleaseDC(NULL, hScreenDC);
        return nullptr;
    }
    if (DEBUG_MODE) {
        printf("SelectObject successful\n");
    }

    // Copy screen to bitmap
    if (!BitBlt(hMemoryDC, 0, 0, *width, *height, hScreenDC, 0, 0, SRCCOPY)) {
        printf("Error: BitBlt failed with error code %lu\n", GetLastError());
        SelectObject(hMemoryDC, hOldBitmap);
        DeleteObject(hBitmap);
        DeleteDC(hMemoryDC);
        ReleaseDC(NULL, hScreenDC);
        return nullptr;
    }
    if (DEBUG_MODE) {
        printf("BitBlt successful\n");
    }

    // Get bitmap info
    BITMAPINFOHEADER bi;
    ZeroMemory(&bi, sizeof(BITMAPINFOHEADER));
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = *width;
    bi.biHeight = -*height;  // Negative height for top-down image
    bi.biPlanes = 1;
    bi.biBitCount = 24;
    bi.biCompression = BI_RGB;

    // Calculate stride (bytes per row, must be DWORD-aligned)
    int stride = (*width * 3 + 3) & ~3;
    if (DEBUG_MODE) {
        printf("Allocating %d bytes for image data\n", stride * *height);
    }

    // Allocate memory for pixel data
    unsigned char* pixels = (unsigned char*)malloc(stride * *height);
    if (!pixels) {
        printf("Error: Failed to allocate memory for pixels\n");
        SelectObject(hMemoryDC, hOldBitmap);
        DeleteObject(hBitmap);
        DeleteDC(hMemoryDC);
        ReleaseDC(NULL, hScreenDC);
        return nullptr;
    }
    if (DEBUG_MODE) {
        printf("Memory allocation successful\n");
    }

    // Get pixel data
    int scanlines = GetDIBits(hMemoryDC, hBitmap, 0, *height, pixels, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    if (scanlines != *height) {
        printf("Error: GetDIBits failed. Expected %d scanlines, got %d. Error code: %lu\n", 
               *height, scanlines, GetLastError());
        free(pixels);
        SelectObject(hMemoryDC, hOldBitmap);
        DeleteObject(hBitmap);
        DeleteDC(hMemoryDC);
        ReleaseDC(NULL, hScreenDC);
        return nullptr;
    }
    if (DEBUG_MODE) {
        printf("GetDIBits successful\n");
    }

    // Cleanup
    SelectObject(hMemoryDC, hOldBitmap);
    DeleteObject(hBitmap);
    DeleteDC(hMemoryDC);
    ReleaseDC(NULL, hScreenDC);

    if (DEBUG_MODE) {
        printf("Screen capture completed successfully\n");
    }
    return pixels;
}
#endif

// Function to get current time in seconds with high precision
float getCurrentTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0f;
}

std::vector<Event> initEvents() {
    // Initialize events
    std::vector<Event> events = {
        {
            "PLANT",
            0.0f,
            "!!!Bomb has been planted!!!",
            "./images/planted_example.png",
            nullptr,
            {1012, 1070},
            {1547, 1149},
            127,
            1000,
            4.0f
        },
        {
            "WON",
            0.0f,
            "!!!Win!!!",
            "./images/win_example.png",
            nullptr,
            {880, 250},
            {1740, 360},
            0,
            5000,
            10.0f
        },
        
        {
            "1KILL",
            0.0f,
            "!!!1st kill!!!",
            "./images/kill1_example.png",
            nullptr,
            {1150, 1200},
            {1325, 1332},
            180,
            2000,
            0.5f
        }
    };

    printf("Loading and processing example images...\n");
    // Load and process example images
    for (auto& event : events) {
        int width, height;
        printf("Loading image: %s\n", event.example_img_path);
        unsigned char* example_img = loadImage(event.example_img_path, &width, &height);
        if (!example_img) {
            printf("Failed to load image: %s\n", event.example_img_path);
            throw std::runtime_error("Failed to load image");
        }

        printf("Loaded image dimensions: %dx%d\n", width, height);

        // Process example image
        int crop_width = event.bottom_right[0] - event.top_left[0];
        int crop_height = event.bottom_right[1] - event.top_left[1];
        event.image = (unsigned char*)malloc(crop_width * crop_height);
        
        if (!event.image) {
            printf("Failed to allocate memory for processed image\n");
            free(example_img);
            throw std::runtime_error("Failed to allocate memory for processed image");
        }

        if (!getCroppedGrayThreshImage(
            example_img,
            event.image,
            width,
            event.top_left,
            event.bottom_right,
            event.threshold
        )) {
            printf("Failed to process example image\n");
            free(example_img);
            free(event.image);
            throw std::runtime_error("Failed to process example image");
        }
    }
    return events;
}

int main() {
    printf("Program starting...\n");
    printf("Initializing CUDA...\n");
    
    if (!initCUDA()) {
        printf("Failed to initialize CUDA. Press Enter to exit...\n");
        getchar();
            return 1;
        }
    
    printf("Screen resolution set to: %dx%d\n", SCREEN_WIDTH, SCREEN_HEIGHT);

    std::vector<Event> events = initEvents();

    printf("Starting main loop. Press ESC to exit.\n");

    const char* check_image_path = "./images/Counter-strike 2 2024.11.17 - 23.00.49.02/PLANT/frame_318.691.png";
    int check_width, check_height;
    unsigned char* check_image = loadImage(check_image_path, &check_width, &check_height);

    Event event = events[0];
        bool is_match = isImgsMatch(
            check_image,
            event.image,
            check_width,
            event.top_left,
            event.bottom_right,
            event.diff_threshold,
            event.threshold
        );
    printf("%s: %s\n", event.name, is_match ? "true" : "false");


    /*
    bool running = true;
    int errorCount = 0;
    const int MAX_ERRORS = 5;

    while (running && errorCount < MAX_ERRORS) {
        try {
            float current_time = getCurrentTime();

            if (DEBUG_MODE) {
                printf("\nAttempting screen capture...\n");
            }
            
            int width, height;
            unsigned char* screenshot = captureScreen(&width, &height);
            if (!screenshot) {
                printf("Screenshot capture failed (%d/%d)\n", ++errorCount, MAX_ERRORS);
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }
            if (DEBUG_MODE) {
                printf("Processing events...\n");
            }
            for (auto& event : events) {
                if (DEBUG_MODE) printf("\nChecking event: %s\n", event.message);
                try {
                    bool is_match = isImgsMatch(
                        screenshot,
                        event.image,
                        width,
                        event.top_left,
                        event.bottom_right,
                        event.diff_threshold,
                        event.threshold
                    );
                    
                    if (is_match) {
                        float time_since_last = current_time - event.last_time;
                        
                        if (time_since_last > event.delay) {
                            printf("%s\n", event.message);
                            event.last_time = current_time;
                        }
                    }
                }
                catch (const std::exception& e) {
                    printf("Event processing error: %s\n", e.what());
                }
            }
            
            free(screenshot);
            errorCount = 0;
            
            if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
                printf("ESC pressed, exiting...\n");
                running = false;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        catch (const std::exception& e) {
            printf("Main loop error: %s (%d/%d)\n", e.what(), ++errorCount, MAX_ERRORS);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
    */
    // Cleanup and exit
    cudaDeviceReset();
    return 0;
}
