
import os
import csv
import cv2

input_csv = '../data/set2/train.csv'
output_csv = '../data/set2/gt.csv'
image_folder = '../data/set2'

csv_outputs = []
visited_files = []

with open(input_csv) as csv_file:

    csv_reader = csv.reader(csv_file)

    for line in csv_reader:

        file_name = line[0]
        print(file_name)

        img = cv2.imread(os.path.join(image_folder, file_name))
        if img is None or file_name in visited_files:
            continue

        visited_files.append(file_name)

        img_height, img_width = img.shape[:2]

        coordinates = line[1:]

        assert len(coordinates) == 8

        xs_ratio = [float(coordinates[2*i])/img_width for i in range(len(coordinates)//2)]
        ys_ratio = [float(coordinates[2*i+1])/img_height for i in range(len(coordinates)//2)]

        coordinate_str = file_name + ','
        for i, (x, y) in enumerate(zip(xs_ratio, ys_ratio)):

            x, y = '{:.8f}'.format(x), '{:.8f}'.format(y)

            if i == 0:
                coordinate_str += '|([' + x + ',' + y + '], '

            elif i == 3:
                coordinate_str += '[' + x + ',' + y + '])|'

            else:
                coordinate_str += '[' + x + ',' + y + '],'

        csv_outputs.append(coordinate_str)

with open(output_csv, 'w+') as csv_file:

    csv_file.writelines([line + '\n' for line in csv_outputs])