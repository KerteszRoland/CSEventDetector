import cv2
import os
from pathlib import Path

def extract_frames(video_path, eventsWithTime):
    # Get video name without extension
    video_name = Path(video_path).stem
    
    # Create output directory
    output_dir = f"./images/{video_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video FPS
    fps = video.get(cv2.CAP_PROP_FPS)
    for eventWithTime in eventsWithTime:
        timestamp = eventWithTime["time"]
        event = eventWithTime["event"]
        # Convert timestamp to frame number
        # Convert MM:SS[:mmm] to seconds if timestamp is in that format
        if isinstance(timestamp, str) and ':' in timestamp:
            parts = timestamp.split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                timestamp = float(minutes) * 60 + float(seconds)
            elif len(parts) == 3:
                minutes, seconds, millis = parts
                timestamp = float(minutes) * 60 + float(seconds) + float(millis)/1000
        
        # Calculate frame numbers for half a second
        start_frame = int((timestamp - 0.5) * fps)
        end_frame = int((timestamp + 0.5) * fps)
        
        # Loop through all frames in this second
        for frame_num in range(start_frame, end_frame):
            # Set video to desired frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
            # Read frame
            ret, frame = video.read()
            if not ret:
                print(f"Error: Could not read frame at timestamp {frame_num/fps:.3f}")
                continue
            
            # Save frame
            frame_timestamp = frame_num / fps
            os.makedirs(os.path.join(output_dir, event), exist_ok=True)
            output_path = os.path.join(output_dir, event, f"frame_{frame_timestamp:.3f}.png")
            cv2.imwrite(output_path, frame)
            print(f"Saved frame at {frame_timestamp:.3f}s to {output_path}")
    
    # Release video
    video.release()

if __name__ == "__main__":
    video_path = "./videos/Counter-strike 2 2024.11.17 - 23.00.49.02.mp4"
    eventsWithTime = [
        {"time": "3:09", "event": "LOST"},
        {"time": "4:21", "event": "LOST"},
        {"time": "5:19", "event": "PLANT"},
        {"time": "5:47", "event": "1KILL"},
        {"time": "5:48", "event": "WON"},
        {"time": "6:40", "event": "PLANT"},
        {"time": "6:49", "event": "WON"},
        {"time": "7:31", "event": "1KILL"},
        {"time": "7:57", "event": "2KILL"},
        {"time": "8:00", "event": "3KILL"},
        {"time": "8:02", "event": "WON"},
        {"time": "10:31", "event": "PLANT"},
        {"time": "11:02", "event": "LOST"},
        {"time": "12:50", "event": "LOST"},
        {"time": "13:42", "event": "PLANT"},
        {"time": "14:01", "event": "WON"},
        {"time": "14:57", "event": "1KILL"},
        {"time": "15:08", "event": "PLANT"},
        {"time": "15:17", "event": "2KILL"},
        {"time": "15:18", "event": "WON"},
        {"time": "15:51", "event": "1KILL"},
        {"time": "15:54", "event": "2KILL"},
        {"time": "16:27", "event": "PLANT"},
        {"time": "16:45", "event": "WON"},
        {"time": "18:22", "event": "LOST"}
  
    ]
    extract_frames(video_path, eventsWithTime)
