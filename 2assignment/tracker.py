'''
Compares the differences between built-in OpenCV trackers.

Notes
-----
Video files are NOT included in this repository (see README)

Arguments
---------
video : The file path of the video to run the trackers on
    Positional, Required

Usage examples
--------------
$python3 tracker.py vidoes/Dusty_snow.mp4
    Runs the trackers on the Dusty_snow video


References
---------
BOOTSING
    [1] https://docs.opencv.org/3.4/d1/d1a/classcv_1_1TrackerBoosting.html#details
    [2] http://www.bmva.org/bmvc/2006/papers/033.pdf

MIL
    [3] https://docs.opencv.org/3.4/d0/d26/classcv_1_1TrackerMIL.html
    [4] https://ieeexplore.ieee.org/document/5674053

KCF
    [5] https://docs.opencv.org/3.4/d2/dff/classcv_1_1TrackerKCF.html
    [6] https://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf

TLD
    [7] https://docs.opencv.org/3.4/dc/d1c/classcv_1_1TrackerTLD.html
    [8] https://pubmed.ncbi.nlm.nih.gov/22156098/

IoU
    [9] https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
'''

import argparse

import cv2
import numpy as np
import pybboxes as pbx
import os.path


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
    parser.add_argument('truth_fp', help='File path to truth')
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video)
    video_name = args.video.split('.')[0]
    video_name = video_name.split('/')[1]

    if not video.isOpened():
        raise Exception(f'Could not find video at {args.video}')

    return video, args.truth_fp, video_name


class Tracker:
    '''
    Tracks objects in provided videos

    Attributes
    ----------
    tracker : cv2[.legacy]?.Tracker[BOOSTING|MIL|KCF|TLD]
    tracker_type : str
    ious : list[float]

    Methods
    -------
    __init__(self, tracker_type string) -> None
        Determines which tracker to use

    read_and_track(self, video: cv2VideoCapture, truth_fp: str) -> None
        Reads the video and tracks the object found therein

    read_truth(self, filepath: str, frame_number: int) -> List[float]
        Reads in the 'truth' (correct bounding box co-ordinates) from the
        filesystem

    calculate_iou(self, truth: list[float], prediction: list[float]) -> float
        [9] Calculates the Intersection over Union (IoU) of a frame to
        quantitatively measure the tracker's performance
    '''

    def __init__(self, tracker_type: str):
        '''
        Determines which tracker to use, options are:
            * BOOSTING
                Real-time object tracking based on the AdaBoost
                algorithm[1][2]
            * MIL
                Multiple Instance Learning, trained to separate the object from
                the background[3][4]
            * KCF
                Kernelized Correlation Filter, "... utilizes properties of
                circulant matrix to enhanace processing speed [5]"[6]
            * TLD
                Tracking, Learning, and Detection, "... follows the object from
                frame to frame, [it] localizes all appearances that have been
                observed so far and corrects [itself] [7]"[8]
        '''

        self.tracker_type = tracker_type

        match tracker_type:
            case 'BOOSTING':
                self.tracker = cv2.legacy.TrackerBoosting_create()
            case 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            case 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            case 'TLD':
                self.tracker = cv2.legacy.TrackerTLD_create()
            case other:
                raise Exception(f'ERROR: {tracker_type} not recognized as a '\
                    'tracker type')

        self.ious = []


    def read_and_track(self, video: cv2.VideoCapture, truth_fp: str):
        '''
        Reads the video and tracks the object found therein

        Parameters
        ----------
        video : cv2.VideoCapture
            The video to be read in

        truth_fp: str
            File path to truth data
        '''

        #Read first frame for setup
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video_ok, frame = video.read()
        if not video_ok:
            raise Exception('ERROR, video.read() did not return ok')

        bbox = cv2.selectROI(frame, False)

        video_ok = self.tracker.init(frame, bbox)
        while True:
            video_ok, frame = video.read()
            if not video_ok:
                print('Video playback aborted')
                break

            frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            truth = self.read_truth(truth_fp, frame_number)
            if not truth:
                continue

            t_p1 = (truth[0], truth[1])
            t_p2 = (truth[2], truth[3])

            tracker_ok, bbox = self.tracker.update(frame)
            # p1 and p2 and the opposite vertexes of the rectangle
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv2.rectangle(frame, t_p1, t_p2, (0,255,0), 2, 1)

            cv2.imshow(self.tracker_type, frame)

            #Flatten Tuples into list
            p = [p1, p2]
            prediction = [i for sub in p for i in sub]

            iou = self.calculate_iou(truth, prediction)
            self.ious.append(iou)

            # Exit if ESC is pressed
            kill = cv2.waitKey(1) & 0xff
            if kill == 27:
                break


        cv2.destroyAllWindows()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)


    def read_truth(self, filepath: str, frame_number: int) -> list[float]:
        '''
        Reads in the 'truth' (correct bounding box co-ordinates) from the
        filesystem

        Parameters
        ----------
        filepath : str
            Filepath to the directory containing the truth co-ordinates

        frame_number : int
            The currently utilized frame of the video containing the object to
            be tracked

        Returns
        -------
        truth : list[float]
            Four co-ordinates representing the correct bounding box of the
            object to be tracked.
        '''
        frame_number = str(frame_number).zfill(6)
        if not os.path.exists(f'{filepath}/frame_{frame_number}.txt'):
            return None
        with open (f'{filepath}/frame_{frame_number}.txt') as f:
            truth = f.readline().strip('\n')

        if not truth:
            return None
        truth = truth.split(' ')
        # All truth files contain a meaningless leading 0, remove it
        truth = truth[1::]
        truth = [float(x) for x in truth]

        video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        truth = pbx.convert_bbox(truth, from_type="yolo", to_type="voc", image_size=(video_width,video_height))

        return truth


    def calculate_iou(self, truth: list[float], prediction: list[float]) -> float:
        '''
        [9] Calculates the Intersection over Union (IoU) of a frame to
        quantitatively measure the tracker's performance. IoU is defined as:

            IoU = (True Positive) / (True Positive + False Positive + False
            Negative)

        Paramters
        ---------
        truth : List[float]
            Four co-ordinates representing the correct bounding box of the
            object to be tracked.

        prediction : List[float]
            Four co-ordinates representing the bounding box predicted by the
            tracker.

        Returns
        ------
        iou : float
            The Intersection over Union
        '''
        truth = np.array(truth, dtype=np.float32)
        prediction = np.array(prediction, dtype=np.float32)

        ix1 = np.maximum(truth[0], prediction[0])
        iy1 = np.maximum(truth[1], prediction[1])
        ix2 = np.minimum(truth[2], prediction[2])
        iy2 = np.minimum(truth[3], prediction[3])

        i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
        i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

        area_of_intersection = i_height * i_width

        truth_height = truth[3] - truth[1] + 1
        truth_width = truth[2] - truth[0] + 1

        prediction_height = prediction[3] - prediction[1] + 1
        prediction_width = prediction[2] - prediction[0] + 1

        area_of_union = truth_height * truth_width + prediction_height * prediction_width - area_of_intersection

        iou = area_of_intersection / area_of_union

        return iou


if __name__ == '__main__':
    video, truth_fp, video_name = setup()
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD']
    trackers = [Tracker(tracker_type) for tracker_type in tracker_types]

    CORRECT_PREDICTION_THRESHOLD = 0.5
    with open('results.txt', 'a') as results:
        results.write(f'{video_name}\n')
        for tracker in trackers:
            tracker.read_and_track(video, truth_fp)
            correct_predictions = [x for x in tracker.ious if x > CORRECT_PREDICTION_THRESHOLD]

            results.write(f'{tracker.tracker_type} IoU mean: '\
                f'{np.mean(tracker.ious):.2f}\n')
            results.write(f'{tracker.tracker_type} IoU standard deviation: '\
                f'{np.std(tracker.ious):.2f}\n')
            results.write(f'{tracker.tracker_type} percentage of predictions with IoU '\
                f'above {CORRECT_PREDICTION_THRESHOLD}: '\
                f'{len(correct_predictions)/len(tracker.ious):.2f}%\n\n')

        results.write('===============================')
