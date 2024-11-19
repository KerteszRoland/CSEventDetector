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
    # Bomb planted
    planted_top_left = (1012, 1070)
    planted_bottom_right = (1547, 1149)
    
    # Win
    win_top_left = (880, 250)
    win_bottom_right = (1740, 360)
    
    # 1st kill 1215
    kill1_top_left = (1150, 1200)
    kill1__bottom_right = (1325, 1332)
    
    in_game = True
    
    last_planted_time = 0
    planted_delay = 4  # seconds
    
    last_win_time = 0
    win_delay = 10  # seconds
    win_threshold = 0
    
    last_kill1_time = 0
    kill1_delay = 0.5  # seconds
    kill1_threshold = 180
    
    example_planted_img = cv2.imread("planted_example.png")
    example_planted_compare_img = get_compare_image(example_planted_img, planted_top_left, planted_bottom_right, 127)

    example_win_img = cv2.imread("win_example.png")
    example_win_compare_img = get_compare_image(example_win_img, win_top_left, win_bottom_right, win_threshold)

    example_kill1_img = cv2.imread("kill1_example.png")
    example_kill1_compare_img = get_compare_image(example_kill1_img, kill1_top_left, kill1__bottom_right, kill1_threshold)
    #cv2.imshow("result", example_win_compare_img)
    #cv2.waitKey(0)

    while True:
        current_time = time.time()
        img = pyautogui.screenshot()
        img = np.array(img)
        if in_game:
            img = cv2.resize(img, (2560, 1440))
        is_planted = is_imgs_match(img, example_planted_compare_img, planted_top_left, planted_bottom_right, 1000, 127)
        is_win = is_imgs_match(img, example_win_compare_img, win_top_left, win_bottom_right, 5000, win_threshold)
        is_kill1 = is_imgs_match(img, example_kill1_compare_img, kill1_top_left, kill1__bottom_right, 2000, kill1_threshold)
        
        if is_planted and (current_time - last_planted_time > planted_delay):
            print("\n!!!Bomb has been planted!!!\n")
            winsound.PlaySound("planted.wav", winsound.SND_ASYNC | winsound.SND_FILENAME)
            last_planted_time = current_time
            
        if is_win and (current_time - last_win_time > win_delay):
            print("\n!!!Win!!!\n")
            winsound.PlaySound("win.wav", winsound.SND_ASYNC | winsound.SND_FILENAME)
            last_win_time = current_time
        
        if is_kill1 and (current_time - last_kill1_time > kill1_delay):
            print("\n!!!1st kill!!!\n")
            winsound.PlaySound("kill.wav", winsound.SND_ASYNC | winsound.SND_FILENAME)
            last_kill1_time = current_time
        #time.sleep(0.25)
        
if __name__ == "__main__":
    main()