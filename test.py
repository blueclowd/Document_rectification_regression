''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import os
import cv2
import csv
import math
from statistics import stdev, mean
import numpy as np
from utils.utils import convert_to_square

import evaluation


def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagePath", default="../058.jpg", help="Path to the document image")
    parser.add_argument("-iv", "--videoPath", default="")
    parser.add_argument("-o", "--outputPath", default="", help="Path to store the result")
    parser.add_argument("-r", "--reportPath", default="", help="Path to store the result")

    parser.add_argument("-rf", "--retainFactor", help="Floating point in range (0,1) specifying retain factor",
                        default="0.85")

    parser.add_argument("-dm", "--documentModel", help="Model for document corners detection",
                        default="../documentModelWell")

    return parser.parse_args()


if __name__ == "__main__":

    args = args_processor()
    corners_extractor = evaluation.corner_extractor.GetCorners(args.documentModel)

    if args.videoPath != '':

        cap = cv2.VideoCapture(args.videoPath)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        warped_size = (1200, int(1200 * 1.4142))
        dst_pts = np.float32(
            [[0, 0], [warped_size[0], 0], [warped_size[0], warped_size[1]], [0, warped_size[1]]]).reshape(-1, 1, 2)

        display_patch = np.zeros((max(frame_width, frame_height), max(frame_width, frame_height)//2, 3), np.uint8)
        cv2.putText(display_patch, text='Warped', org=(550, 350), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=max(frame_width, frame_height) * 0.001, color=(255, 255, 255),
                    lineType=cv2.LINE_AA, thickness=int(max(frame_width, frame_height) * 0.002))

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_writer = cv2.VideoWriter(os.path.join(args.outputPath, 'single.avi'), fourcc, 20.0, (int(max(frame_width,frame_height) * 1.5), max(frame_width, frame_height)))

        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to square image
            square, padding_dist, padding_axis = convert_to_square(frame, (128, 128, 128))
            square = cv2.transpose(square)
            square = cv2.flip(square, flipCode=1)

            corners = corners_extractor.get(square)

            # Warping
            src_pts = np.float32([corners]).reshape(-1, 1, 2)
            homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            aligned_img = cv2.warpPerspective(square, homography_matrix, warped_size)

            display_patch[400:400 + warped_size[1], 200:200+warped_size[0], :] = aligned_img

            # Overlay
            cv2.polylines(square, pts=np.array([corners], np.int32), isClosed=True, color=(255, 255, 255),
                          thickness=int(0.005 * frame_width), lineType=cv2.LINE_AA)

            for i, corner in enumerate(corners):
                cv2.circle(square, center=corner, radius=int(0.01 * frame_width), color=(255, 255, 0), thickness=-1,
                           lineType=cv2.LINE_AA)
                cv2.putText(square, text=str(i), org=corner, fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=int(0.002 * frame_width), color=(0, 255, 0), lineType=cv2.LINE_AA,
                            thickness=int(0.003 * frame_width))

            square = np.hstack((square, display_patch))

            video_writer.write(square)
            square = cv2.resize(square, (768, 512))

            cv2.imshow('frame', square)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        video_writer.release()


    elif args.imagePath != '':

        image_names = os.listdir(args.imagePath)
        image_names = [image_name for image_name in image_names if image_name.endswith(('.jpg', '.png'))]

        # Load ground truth
        gts = {}
        csv_path = os.path.join(args.imagePath, 'gt.csv')
        with open(csv_path) as csv_file:

            csv_reader = csv.reader(csv_file)

            for line in csv_reader:

                image_name = line[0]
                img = cv2.imread(os.path.join(args.imagePath, image_name))
                img_height, img_width = img.shape[:2]

                gts[line[0]] = {'tl': np.array([float(line[1]), float(line[2])]), 'tr': np.array([float(line[3]), float(line[4])]),
                                'br': np.array([float(line[5]), float(line[6])]), 'bl': np.array([float(line[7]), float(line[8])])}

        # Prediction
        predictions = {}
        for image_name in image_names:

            img = cv2.imread(os.path.join(args.imagePath, image_name))
            img_height, img_width = img.shape[:2]

            corners = corners_extractor.get(img)

            prediction = {'tl': np.array(corners[0]), 'tr': np.array(corners[1]), 'br': np.array(corners[2]), 'bl': np.array(corners[3])}
            predictions[image_name] = prediction

            cv2.polylines(img, pts=np.array([corners], np.int32), isClosed=True, color=(255, 255, 255), thickness=int(0.005*img_width), lineType=cv2.LINE_AA)

            for i, corner in enumerate(corners):
                cv2.circle(img, center=corner, radius=int(0.01*img_width), color=(255, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
                cv2.putText(img, text=str(i), org=(corner), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=int(0.002*img_width), color=(0, 255, 0), lineType=cv2.LINE_AA, thickness=int(0.003*img_width))

            cv2.imwrite(os.path.join(args.outputPath, image_name), img)

        # Evaluation(Root mean square error)
        rmses = {}
        for file_name, coordinates in predictions.items():

            gt = gts[file_name]
            rmse = math.sqrt(
                mean(np.concatenate(((coordinates['tl'] - gt['tl']) ** 2, (coordinates['tr'] - gt['tr']) ** 2,
                                     (coordinates['bl'] - gt['bl']) ** 2, (coordinates['br'] - gt['br']) ** 2))))

            rmses[file_name] = rmse

        rmse_mean = mean([value for file_name, value in rmses.items()])
        rmse_std = stdev([value for file_name, value in rmses.items()])

        print('RMSR(mean): {}'.format(rmse_mean))
        print('RMSR(std): {}'.format(rmse_std))

        with open(args.reportPath, 'w', newline='') as csv_file:

            csv_writer = csv.writer(csv_file)
            for file_name, rmse in rmses.items():

                csv_writer.writerow([file_name, rmse])

            csv_writer.writerow(['Mean', rmse_mean])
            csv_writer.writerow(['Std', rmse_std])

        print('Generated RMSE report.')
        