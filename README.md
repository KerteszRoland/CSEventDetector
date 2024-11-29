# Video Game Event Detection Based on Incoming Video Stream

I built a system that can detect events on the screen while playing a game by using image processing and CUDA programming.

The system is capable of running in real time and in my setup, I can run it while playing a game.
I gathered example images from Counter Strike 2, but you can use it for anything that is based on an event that appears on the screen.

[Slides](https://drive.google.com/drive/folders/1ZZykoV-DxckxmGlggyy43D2z74HiIiE0?usp=sharing)

## Code Structure

### Main Function

- Checks for CUDA
- Initializes events
- Runs in selected mode

### InitEvents Function

- You can set your events in here with all the details of the event
- It loads and preprocesses the image for later comparison

## Available Modes

### Test Image Mode

- Arguments are the following: `<event_name> <image_path>`
- It returns true if the event occurred in the image

### Test Images Mode

- Arguments are the following: `<event_name> <test_images_dir>`
- It tests all images in the directory against the event
- Prints out which images did not match and their difference values
- Allows you to pick which non-matching images should be added as new good examples

### Real Time Mode

- No additional arguments needed
- Captures screen in real time
- Compares screen against all events
- If event matches and enough time passed since last match:
  - Prints event message
  - Plays sound if enabled
- Press ESC to exit

### Help Mode

- Displays how to use the program
- Usage: mask.exe <mode> [arguments]
  - Modes:
    - test_image <event_name> <image_path>
    - test_images <event_name> <test_images_dir>
    - real_time
    - help

## Technical Details

### Event Configuration

Events are defined with the following parameters:

- `last_time`: Tracks when event was last triggered (initialized to 0)
- `sound`: Path to sound file to play when event occurs
- `message`: Text message to display when event occurs
- `example_img`: Path to example image to match against
- `image`: Preprocessed example image (initialized as None)
- `top_left`: (x,y) coordinates of top-left corner of detection region
- `bottom_right`: (x,y) coordinates of bottom-right corner of detection region
- `threshold`: Threshold value for binary image conversion (0-255)
- `diff_threshold`: Maximum allowed difference between images to count as a match
- `delay`: Minimum seconds between event triggers (real-time mode only)

### Image Processing

Image processing is accelerated using CUDA on the GPU

1. Crops the screen capture to the specified region
2. Converts to grayscale
3. Applies binary threshold if threshold > 0
4. Compares against preprocessed example image
5. Calculates absolute difference between images
6. Triggers event if difference is below threshold
