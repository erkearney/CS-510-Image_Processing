'''
Compares the differences between built-in OpenCV trackers.

Notes
-----
Video files are NOT included in this repository (see README)


Arguments
---------
video : The file path of the video to run the trackers on
    Positional, Required
-d --debug : Enter debug mode

Usage examples
--------------
$python3 tracker.py vidoes/Dusty_snow.mp4
    Runs the trackers on the Dusty_snow video

'''

import argparse

import cv2


def setup() -> cv2.VideoCapture:
    '''
    Parses arguments and reads in video

    Returns
    -------
    video : cv2.VideoCapture
        The video to be assessed

    Raises
    ------
    Exception
        If the specified video is not found, raise a generic Exception
    '''
    parser = argparse.ArgumentParser(description='Compare the performance of '\
        'different trackers built into OpenCV')
    parser.add_argument('video', help='File path to video')
    parser.add_argument('-d', '--debug', action='store_true', help='Enter '\
        'debug mode')
    args = parser.parse_args()
    if args.debug:
        print(f'args: {args}')

    video = cv2.VideoCapture(args.video)

    if not video.isOpened():
        raise Exception(f'Could not find video at {args.video}')

    return video


def read_and_track(video: cv2.VideoCapture):
    '''
    Reads the video and tracks the object found therein

    Parameters
    ----------
    video : cv2.VideoCapture
        The video to be read in


    Raises
    ------
    Exception
        If video cannot be properly read, raise a generic Exception

        TODO: Fix Exception being raised when video naturally ends
    '''
    #Read first frame for setup
    video_ok, frame = video.read()
    if not video_ok:
        raise Exception(f'ERROR, video.read() did not return ok')

    bbox = cv2.selectROI(frame, False)
    tracker = cv2.TrackerMIL_create()
    video_ok = tracker.init(frame, bbox)

    while True:
        video_ok, frame = video.read()
        if not video_ok:
            raise Exception(f'ERROR, video.read() did not return ok')

        tracker_ok, bbox = tracker.update(frame)
        # p1 and p2 and the opposite vertexes of the rectangle
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        cv2.imshow('MIL', frame)

        #Exit if ESC is pressed
        kill = cv2.waitKey(1) & 0xff
        if kill == 27:
            break


if __name__ == '__main__':
    video = setup()
    read_and_track(video)
