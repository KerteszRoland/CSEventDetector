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
    const int top_left[2];
    const int bottom_right[2];
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

void saveImage(const char* filename, const unsigned char* image, int width, int height, bool is_grayscale=true) {

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

__global__ void grayscaleThresholdDifferenceKernel(
    const unsigned char* input,
    unsigned char* output,
    const int width,
    const int height,
    const unsigned char threshold,
    const unsigned char* example_img,
    int* d_difference
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    const int x = idx % width;
    const int y = idx / width;
    const int rgb_idx = (y * width + x) * 3;
    
    // RGB to grayscale conversion
    const unsigned char gray = 0.299f * input[rgb_idx] + 
                      0.587f * input[rgb_idx + 1] + 
                      0.114f * input[rgb_idx + 2];
    
    // Thresholding
    const unsigned char thresholded = (threshold > 0 && gray > threshold) ? 255 : 0;
    output[idx] = thresholded;
    //Calculate difference
    if(d_difference == nullptr || example_img == nullptr) return;

    const unsigned char diff = abs(thresholded - example_img[idx]);
    atomicAdd(d_difference, diff);
}

void cropImage(const unsigned char* input, unsigned char* output, const int full_width, const int* top_left, const int* bottom_right) {
    int crop_width = bottom_right[0] - top_left[0];
    int crop_height = bottom_right[1] - top_left[1];

    // Copy cropped region
    for (int y = 0; y < crop_height; y++) {
        for (int x = 0; x < crop_width; x++) {
            int src_idx = ((y + top_left[1]) * full_width + (x + top_left[0])) * 3;
            int dst_idx = (y * crop_width + x) * 3;
            
            // Copy RGB channels
            output[dst_idx] = input[src_idx];        // R
            output[dst_idx + 1] = input[src_idx + 1];// G 
            output[dst_idx + 2] = input[src_idx + 2];// B
        }
    }
}

// Function implementation
void getCroppedGrayThreshImage(
    const unsigned char* input,
    unsigned char* output,
    const int full_width,
    const int* top_left,
    const int* bottom_right,
    const unsigned char threshold
) {
    int crop_width = bottom_right[0] - top_left[0];
    int crop_height = bottom_right[1] - top_left[1];
    int size = crop_width * crop_height;    
    
    unsigned char* processed_img = (unsigned char*)malloc(size * 3);  // Allocate for RGB cropped image
    cropImage(input, processed_img, full_width, top_left, bottom_right);

    // CUDA

    unsigned char* d_processed_img;
    cudaMalloc(&d_processed_img, size*3);
    cudaMemcpy(d_processed_img, processed_img, size*3, cudaMemcpyHostToDevice);

    unsigned char* d_output;
    cudaMalloc(&d_output, size);

    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;
    grayscaleThresholdDifferenceKernel<<<numBlocks, blockSize>>>(
        d_processed_img,
        d_output,
        crop_width,
        crop_height,
        threshold,
        nullptr,
        nullptr
    );
   
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_processed_img);
    cudaFree(d_output);

    free(processed_img);
}

bool isImgsMatch(
    const unsigned char* img,
    const unsigned char* example_img,
    const int full_width,
    const int* top_left,
    const int* bottom_right,
    const int diff_threshold,
    const unsigned char threshold
) {
    int crop_width = bottom_right[0] - top_left[0];
    int crop_height = bottom_right[1] - top_left[1];
    int size = crop_width * crop_height;    
    
    unsigned char* processed_img = (unsigned char*)malloc(size * 3);  // Allocate for RGB cropped image
    cropImage(img, processed_img, full_width, top_left, bottom_right);

    unsigned char* output = (unsigned char*)malloc(size);  // Allocate for grayscale cropped image
    int difference = 0;

    // CUDA

    int* d_difference;
    cudaMalloc(&d_difference, sizeof(int));
    cudaMemset(d_difference, 0, sizeof(int));

    unsigned char* d_processed_img;
    cudaMalloc(&d_processed_img, size*3);
    cudaMemcpy(d_processed_img, processed_img, size*3, cudaMemcpyHostToDevice);

    unsigned char* d_example_img;
    cudaMalloc(&d_example_img, size);
    cudaMemcpy(d_example_img, example_img, size, cudaMemcpyHostToDevice);

    unsigned char* d_output;
    cudaMalloc(&d_output, size);

    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;
    grayscaleThresholdDifferenceKernel<<<numBlocks, blockSize>>>(
        d_processed_img,
        d_output,
        crop_width,
        crop_height,
        threshold,
        d_example_img,
        d_difference
    );

    cudaMemcpy(&difference, d_difference, sizeof(int), cudaMemcpyDeviceToHost);    
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_processed_img);
    cudaFree(d_example_img);
    cudaFree(d_difference);
    cudaFree(d_output);


    // Save debug images
    char debug_filename[256];
    snprintf(debug_filename, sizeof(debug_filename), "./images/debug/%s.png", "example_img");
    saveImage(debug_filename, example_img, crop_width, crop_height, true);
    saveImage("./images/debug/processed.png", output, crop_width, crop_height, true);

    free(output);
    free(processed_img);
    
    difference = difference / 255;
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
        printf("Loaded image dimensions: %dx%d\n", width, height);

        // Process example image
        int crop_width = event.bottom_right[0] - event.top_left[0];
        int crop_height = event.bottom_right[1] - event.top_left[1];
        int size = crop_width * crop_height;
        event.image = (unsigned char*)malloc(size); // allocate croppedGrayThreshImage

        getCroppedGrayThreshImage(
            example_img,
            event.image,
            width,
            event.top_left,
            event.bottom_right,
            event.threshold
        );
    }
    return events;
}

bool isEventMatch(
    const unsigned char* img,
    const Event& event
){
    return isImgsMatch(
        img,
        event.image,
        SCREEN_WIDTH,
        event.top_left,
        event.bottom_right,
        event.diff_threshold,
        event.threshold
    );
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
    //const char* check_image_path = "./images/planted_example.png";
    int check_width, check_height;
    unsigned char* check_image = loadImage(check_image_path, &check_width, &check_height);

    Event event = events[0];
    bool is_match = isEventMatch(check_image, event);
    if (is_match) {
        printf("%s\n", event.message);
    }

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
    for (auto& event : events) {
        free(event.image);
    }

    cudaDeviceReset();
    return 0;
}
