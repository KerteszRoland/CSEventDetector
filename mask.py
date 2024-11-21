import pyautogui
import cv2
import numpy as np
import time
import winsound

def get_cropped_gray_thresh_image(img, top_left, bottom_right, threshold=127):
    img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if threshold > 0:
        img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
    return img
    

def is_imgs_match(img, example_img_compare, top_left, bottom_right, diff_threshold=2000, threshold=127):
    img = get_cropped_gray_thresh_image(img, top_left, bottom_right, threshold)
    difference = cv2.absdiff(example_img_compare, img)
    difference_value = np.sum(difference) / 255
    
    #print(f"Difference value: {difference_value}")
    #cv2.imshow("train", planted)
    #cv2.imshow("test", planted_test)
    #cv2.imshow("result", difference)
    #cv2.waitKey(1)
    if(difference_value < diff_threshold):
        #print(difference_value)
        return True
    
    return False


def main():
    events = {
        "planted":{
            "last_time": 0,
            "sound": "./sounds/planted.wav",
            "message": "!!!Bomb has been planted!!!",
            "example_img": "./images/planted_example.png",
            "image": None,
            "top_left": (1012, 1070),
            "bottom_right": (1547, 1149),
            "threshold": 127,
            "diff_threshold": 1000,
            "delay": 4
        },
        "win":{
            "last_time": 0,
            "sound": "./sounds/win.wav",
            "message": "!!!Win!!!",
            "example_img": "./images/win_example.png",
            "image": None,
            "top_left": (880, 250),
            "bottom_right": (1740, 360),
            "threshold": 0,
            "diff_threshold": 5000,
            "delay": 10
        },
        "kill1":{
            "last_time": 0,
            "sound": "./sounds/kill.wav",
            "message": "!!!1st kill!!!",
            "example_img": "./images/kill1_example.png",
            "image": None,
            "top_left": (1150, 1200),
            "bottom_right": (1325, 1332),
            "threshold": 180,
            "diff_threshold": 2000,
            "delay": 0.5
        }
    }
    

    for event_name, event in events.items():
        top_left = event["top_left"]
        bottom_right = event["bottom_right"]
        threshold = event["threshold"]
        example_img_path = event["example_img"]

        example_img = cv2.imread(example_img_path)
        if example_img.shape != (1440, 2560, 3):
            example_img = cv2.resize(example_img, (2560, 1440))
        compare_img = get_cropped_gray_thresh_image(example_img, top_left, bottom_right, threshold)
        events[event_name]["image"] = compare_img

    #cv2.imshow("result", example_win_compare_img)
    #cv2.waitKey(0)

    while True:
        current_time = time.time()
        img = pyautogui.screenshot()
        img = np.array(img)
        if img.shape != (1440, 2560, 3):
            img = cv2.resize(img, (2560, 1440))
        
        for event_name, event in events.items():
            event_image = event["image"]
            top_left = event["top_left"]
            bottom_right = event["bottom_right"]
            diff_threshold = event["diff_threshold"]
            threshold = event["threshold"]
            delay = event["delay"]
            sound = event["sound"]
            message = event["message"]
            
            is_match = is_imgs_match(img, event_image, top_left, bottom_right, diff_threshold, threshold)
            enough_time_passed = (current_time - event["last_time"] > delay)
            if is_match and enough_time_passed:
                print(message)
                events[event_name]["last_time"] = current_time
                winsound.PlaySound(sound, winsound.SND_ASYNC | winsound.SND_FILENAME)
            
        
if __name__ == "__main__":
    main()