import cv2
import numpy as np
import sys

BACKGROUND_VIDEO_PATH = "/home/lucas/Inatel/C209-L1/Trabalho_final/ocean.mp4"

def resize(dst, img):
    try:
        width = img.shape[1]
        height = img.shape[0]
        dim = (width, height)
        resized = cv2.resize(dst, dim)
        return resized
    except Exception as e:
        print("Error resizing image")
        print(e)
        return dst


def setup() -> (cv2.VideoCapture, cv2.VideoCapture, np.ndarray, int):
    video = cv2.VideoCapture(-1)
    oceanVideo = cv2.VideoCapture(BACKGROUND_VIDEO_PATH)
    success, ref_img = video.read()
    flag = 0
    return video, oceanVideo, ref_img, flag


def check_background_has_opened(oceanVideo: cv2.VideoCapture):
    if not oceanVideo.isOpened():
        print("Erro ao abrir o video de background")

        sys.exit()


def create_mask(img: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
    diff1 = cv2.subtract(img, ref_img)
    diff2 = cv2.subtract(ref_img, img)
    diff = diff1+diff2
    diff[abs(diff) < 13.0] = 0
    gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray[np.abs(gray) < 10] = 0
    fgmask = gray.astype(np.uint8)
    fgmask[fgmask > 0] = 255
    return fgmask


def combine_images(fgimg: np.ndarray, bgimg: np.ndarray) -> np.ndarray:
    dst = cv2.add(bgimg, fgimg)
    cv2.imshow('Remoção de fundo em tempo real', dst)


def main_loop(video: cv2.VideoCapture, oceanVideo: cv2.VideoCapture, ref_img: np.ndarray, flag: int):
    while (1):
        success, img = video.read()
        success2, bg = oceanVideo.read()
        bg = resize(bg, ref_img)
        if flag == 0:
            ref_img = img

        # create a mask
        fgmask = create_mask(img, ref_img)
        # invert the mask
        fgmask_inv = cv2.bitwise_not(fgmask)
        # use the masks to extract the relevant parts from FG and BG
        fgimg = cv2.bitwise_and(img, img, mask=fgmask)
        bgimg = cv2.bitwise_and(bg, bg, mask=fgmask_inv)
        # combine both the BG and the FG images
        combine_images(fgimg, bgimg)
        key = cv2.waitKey(5) & 0xFF
        if ord('q') == key:
            flag = - 1
        elif ord('d') == key:
            flag = 1
            print("Background Captured")
        elif ord('r') == key:
            flag = 0
            print("Ready to Capture new Background")

    cv2.destroyAllWindows()
    video.release()


def main():
    video, oceanVideo, ref_img, flag = setup()
    check_background_has_opened(oceanVideo)
    main_loop(video, oceanVideo, ref_img, flag)


if __name__ == "__main__":
    main()
