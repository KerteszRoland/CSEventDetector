import pyautogui
import cv2
import numpy as np
import keyboard
import time
import winsound

def get_compare_image(img, top_left, bottom_right, threshold=127):
    img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if threshold > 0:
        img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
    return img
    

def is_imgs_match(img, example_img_compare, top_left, bottom_right, diff_threshold=2000, threshold=127):
    planted_test = get_compare_image(img, top_left, bottom_right, threshold)
    difference = cv2.absdiff(example_img_compare, planted_test)
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
            "name": "planted",
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
            "name": "win",
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
            "name": "kill1",
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
    
    in_game = True
    for event_name, event in events.items():
        
        example_img = cv2.imread(event["example_img"])
        compare_img = get_compare_image(example_img, event["top_left"], event["bottom_right"], event["threshold"])
        events[event_name]["image"] = compare_img

    #cv2.imshow("result", example_win_compare_img)
    #cv2.waitKey(0)

    while True:
        current_time = time.time()
        img = pyautogui.screenshot()
        img = np.array(img)
        if in_game:
            img = cv2.resize(img, (2560, 1440))
            
        for event_name, event in events.items():
            image = event["image"]
            top_left = event["top_left"]
            bottom_right = event["bottom_right"]
            diff_threshold = event["diff_threshold"]
            threshold = event["threshold"]
            delay = event["delay"]
            
            is_match = is_imgs_match(img, image, top_left, bottom_right, diff_threshold, threshold)
            enough_time_passed = (current_time - event["last_time"] > delay)
            if is_match and enough_time_passed:
                print(event["message"])
                events[event["name"]]["last_time"] = current_time
                winsound.PlaySound(event["sound"], winsound.SND_ASYNC | winsound.SND_FILENAME)
            
        
if __name__ == "__main__":
    main()