#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <vector>
#include <stdexcept>


#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")  // Link with winmm.lib
#endif


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define SCREEN_WIDTH 2560
#define SCREEN_HEIGHT 1440 

#define DEBUG_MODE false
#define NO_SOUNDS true

struct Event {
    const char* name;
    float last_time;
    const char* message;
    const char* example_img_path;
    const char* sound_path;
    unsigned char* image;
    const int top_left[2];
    const int bottom_right[2];
    const unsigned char threshold;
    const int diff_threshold;
    const float delay;
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
 }
}

#ifdef _WIN32
unsigned char* captureScreen(int* width, int* height) {
    // Use the constants instead of GetSystemMetrics
    *width = SCREEN_WIDTH;   // 2560
    *height = SCREEN_HEIGHT; // 1440

    // Create device context and bitmap
    HDC hScreenDC = GetDC(NULL);
    if (!hScreenDC) {
        printf("Error: GetDC failed with error code %lu\n", GetLastError());
        return nullptr;
    }

    HDC hMemoryDC = CreateCompatibleDC(hScreenDC);
    if (!hMemoryDC) {
        printf("Error: CreateCompatibleDC failed with error code %lu\n", GetLastError());
        ReleaseDC(NULL, hScreenDC);
        return nullptr;
    }

    HBITMAP hBitmap = CreateCompatibleBitmap(hScreenDC, *width, *height);
    if (!hBitmap) {
        printf("Error: CreateCompatibleBitmap failed with error code %lu\n", GetLastError());
        DeleteDC(hMemoryDC);
        ReleaseDC(NULL, hScreenDC);
        return nullptr;
    }

    HBITMAP hOldBitmap = (HBITMAP)SelectObject(hMemoryDC, hBitmap);
    if (!hOldBitmap) {
        printf("Error: SelectObject failed with error code %lu\n", GetLastError());
        DeleteObject(hBitmap);
        DeleteDC(hMemoryDC);
        ReleaseDC(NULL, hScreenDC);
        return nullptr;
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

    // Cleanup
    SelectObject(hMemoryDC, hOldBitmap);
    DeleteObject(hBitmap);
    DeleteDC(hMemoryDC);
    ReleaseDC(NULL, hScreenDC);

    return pixels;
}

void playSound(const char* sound_path) {
    if (!PlaySoundA(sound_path, NULL, SND_FILENAME | SND_ASYNC)) {
        printf("Error playing sound: %lu\n", GetLastError());
    }
}
#endif


float getCurrentTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0f;
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
    const unsigned char threshold,
    int* output_difference = nullptr
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


    if(DEBUG_MODE){
        // Save debug images
        char debug_filename[256];
        snprintf(debug_filename, sizeof(debug_filename), "./images/debug/%s.png", "example_img");
        saveImage(debug_filename, example_img, crop_width, crop_height, true);
        saveImage("./images/debug/processed.png", output, crop_width, crop_height, true);
    }

    free(output);
    free(processed_img);
    
    difference = difference / 255;
    if (output_difference != nullptr) {
        *output_difference = difference;
    }
    if (DEBUG_MODE) printf("%d %d\n", difference, diff_threshold);
    return difference < diff_threshold;
}

std::vector<Event> initEvents() {
    // Initialize events
    std::vector<Event> events = {
        {
            "PLANT",
            0.0f,
            "!!!Bomb has been planted!!!",
            "./images/planted_example.png",
            "./sounds/planted.wav",
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
            "./images/won_example.png",
            "./sounds/won.wav",
            nullptr,
            {880, 250},
            {1740, 360},
            127,
            3000,
            10.0f
        },
        {
            "1KILL",
            0.0f,
            "!!!1st kill!!!",
            "./images/kill1_example.png",
            "./sounds/kill1.wav",
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
    unsigned char* example_img = nullptr;
    int width, height;
    for (auto& event : events) {
        printf("Loading image: %s\n", event.example_img_path);
        example_img = loadImage(event.example_img_path, &width, &height);
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
    free(example_img);
    return events;
}

bool isEventMatch(
    const unsigned char* img,
    const Event& event,
    int* output_difference = nullptr
){
    return isImgsMatch(
        img,
        event.image,
        SCREEN_WIDTH,
        event.top_left,
        event.bottom_right,
        event.diff_threshold,
        event.threshold,
        output_difference
    );
}

bool testEventWithImage(const std::vector<Event>& events, const char* event_name, const std::string& image_path) {
    // Test event matching on a single image
    const Event& event = *std::find_if(events.begin(), events.end(), 
        [event_name](const Event& e) { return strcmp(e.name, event_name) == 0; });

    int width, height;
    unsigned char* test_image = loadImage(image_path.c_str(), &width, &height);
        if (!test_image) {
        printf("Failed to load image: %s\n", image_path.c_str());
        free(test_image);
        throw std::runtime_error("Failed to load image");
    }

    bool matched = isEventMatch(test_image, event);
    free(test_image);
    
    printf("Matched: %s\n", matched ? "MATCHED" : "NOT MATCHED");

    return matched;
}

std::pair<std::vector<std::string>, std::vector<int>> testEventWithImages(const std::vector<Event>& events, const char* event_name, const std::string& test_dir_prefix) {
    // Test event matching on a directory of images
    const Event& event = *std::find_if(events.begin(), events.end(), 
        [event_name](const Event& e) { return strcmp(e.name, event_name) == 0; });

    std::string test_dir = test_dir_prefix + event_name + "/";
    std::vector<std::string> image_files;
    
    WIN32_FIND_DATA findData;
    HANDLE hFind = FindFirstFile((test_dir + "*.png").c_str(), &findData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            image_files.push_back(test_dir + findData.cFileName);
        } while (FindNextFile(hFind, &findData));
        FindClose(hFind);
    }

    printf("Found %zu images to test\n", image_files.size());

    std::vector<std::string> notMatchedPaths;
    std::vector<int> notMatchedDifferences;

    int img_width, img_height;
    unsigned char* test_image = nullptr;
    for (const auto& image_path : image_files) {
        test_image = loadImage(image_path.c_str(), &img_width, &img_height);
        
        if (!test_image) {
            printf("Failed to load image: %s\n", image_path.c_str());
            continue;
        }
        int* difference = (int*)malloc(sizeof(int));
        bool matched = isEventMatch(test_image, event, difference);
        if(!matched) {
            notMatchedPaths.push_back(image_path);
            notMatchedDifferences.push_back(*difference);
        }
        free(difference);
        free(test_image);
    }

    printf("Found %zu not matched images\n", notMatchedPaths.size());
    for (size_t i = 0; i < notMatchedPaths.size(); i++) {
        printf("%s (%d)\n", notMatchedPaths[i].c_str(), notMatchedDifferences[i]);
    }

    return std::make_pair(notMatchedPaths, notMatchedDifferences);
}

void pickNewGoodExamples(const char* event_name, const std::vector<std::string>& notMatchedPaths, const std::vector<int>& notMatchedDifferences) {
    std::vector<int> filteredDifferences;
    std::vector<std::string> filteredPaths;
    
    // Filter out very similar images
    for (size_t i = 0; i < notMatchedDifferences.size(); i++) {
        bool is_unique_with_margin = true;
        for (size_t j = 0; j < filteredDifferences.size(); j++) {
            if (abs(notMatchedDifferences[i] - filteredDifferences[j]) < 100) {
                is_unique_with_margin = false;
                break;
            }
        }
        if (is_unique_with_margin) {
            filteredDifferences.push_back(notMatchedDifferences[i]);
            filteredPaths.push_back(notMatchedPaths[i]);
        }
    }
    
    // Create examples directory if it doesn't exist
    std::string examples_dir = "./images/examples/" + std::string(event_name) + "/";
    CreateDirectory("./images", NULL);
    CreateDirectory("./images/examples", NULL);
    CreateDirectory(examples_dir.c_str(), NULL);

    printf("\nManual filtering of not matched images:\n");
    printf("Press ENTER to keep image, BACKSPACE to skip\n");


    char fullPath[MAX_PATH];
    for (size_t i = 0; i < filteredPaths.size(); i++) {
        const auto& path = filteredPaths[i];
        const std::string window_title = path.substr(path.find_last_of("/\\") + 1);
        
        // Get the full path
        if (GetFullPathNameA(path.c_str(), MAX_PATH, fullPath, nullptr) == 0) {
            printf("Error getting full path for: %s\n", path.c_str());
            continue;
        }
        
        // Open the image using the default image viewer
        HINSTANCE result = ShellExecuteA(
            NULL,           // No parent window
            "open",         // Operation
            fullPath,       // File path
            NULL,          // Parameters
            NULL,          // Working directory
            SW_SHOWMAXIMIZED  // Show window maximized (fullscreen)
        );
        
        printf("Reviewing: %s (%zu/%zu)\n", path.c_str(), i + 1, filteredPaths.size());
        
        bool validKey = false;
        while (!validKey) {
            if (GetAsyncKeyState(VK_RETURN) & 0x8000) {  // ENTER key
                // Copy file to examples directory
                std::string filename = path.substr(path.find_last_of("/\\") + 1);
                std::string dest_path = "./images/examples/" + std::string(event_name) + "/" + filename;
                if (CopyFileA(fullPath, dest_path.c_str(), FALSE)) {
                    printf("Copied to: %s\n", dest_path.c_str());
                } else {
                    printf("Failed to copy file. Error: %lu\n", GetLastError());
                }
                validKey = true;
                Sleep(200);
            }
            else if (GetAsyncKeyState(VK_BACK) & 0x8000) {  // BACKSPACE key
                printf("Skipped\n");
                validKey = true;
                Sleep(200);
            }
            Sleep(10);
        }

        // Close the image viewer window
        HWND hwnd = FindWindowA(NULL, window_title.c_str());
        if (hwnd != NULL) {
            PostMessage(hwnd, WM_CLOSE, 0, 0);
        }
    }

}

void detectEventsOnScreen(std::vector<Event>& events) {
    bool running = true;
    int errorCount = 0;
    const int MAX_ERRORS = 5;
    int width, height;
    unsigned char* screenshot = nullptr;

    printf("Starting main loop. Press ESC to exit.\n");
    while (running && errorCount < MAX_ERRORS) {
        try {
            float current_time = getCurrentTime();
            try{
                screenshot = captureScreen(&width, &height);
            }
            catch(const std::exception& e){
                free(screenshot);
                throw e;
            }

            for (auto& event : events) {
                bool is_match = isEventMatch(
                    screenshot,
                    event
                );
                
                if (is_match) {
                    float time_since_last = current_time - event.last_time;
                    if (time_since_last > event.delay) {
                        printf("%s\n", event.message);
                        if (!NO_SOUNDS && event.sound_path != nullptr) {
                           playSound(event.sound_path);
                        }
                        event.last_time = current_time;
                    }
                }
            }
          
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
    free(screenshot);
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

    //testEventWithImage(events, "PLANT", "./images/Counter-strike 2 2024.11.17 - 23.00.49.02/PLANT/frame_318.641.png");
    //std::pair<std::vector<std::string>, std::vector<int>> notMatched = testEventWithImages(events, "1KILL", "./images/Counter-strike 2 2024.11.17 - 23.00.49.02/");
    //pickNewGoodExamples("1KILL", notMatched.first, notMatched.second);
    detectEventsOnScreen(events);

    // Cleanup and exit
    for (auto& event : events) {
        free(event.image);
    }

    cudaDeviceReset();
    return 0;
}
