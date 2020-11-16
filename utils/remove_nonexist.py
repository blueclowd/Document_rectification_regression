
import os
import csv

input_csv = '../data/test/gt.csv'
output_csv = '../data/test/gt_.csv'

image_folder = '../data/test'
visited_files = []

csv_outputs = []
with open(input_csv) as csv_file:

    csv_reader = csv.reader(csv_file)

    for line in csv_reader:

        file_name = line[0]

        if file_name in visited_files:
            continue

        else:
            visited_files.append(file_name)

        if os.path.exists(os.path.join(image_folder, file_name)):

            csv_outputs.append(line)

csv_outputs.sort(key=lambda line: line[0])

with open(output_csv, 'w') as csv_file:

    csv_writer = csv.writer(csv_file)

    csv_writer.writerows(csv_outputs)