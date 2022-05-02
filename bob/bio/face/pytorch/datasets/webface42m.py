#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import csv
import os

import numpy as np

from torch.utils.data import Dataset

import bob.io.base

# from bob.bio.face.database import MEDSDatabase, MorphDatabase
from bob.extension import rc
from bob.extension.download import get_file, search_file


class WebFace42M(Dataset):
    """
    Pytorch Daset for the WebFace42M dataset mentioned in


    .. latex::

        @inproceedings {zhu2021webface260m,
                title=  {WebFace260M: A Benchmark Unveiling the Power of Million-scale Deep Face Recognition},
                author=  {Zheng Zhu, Guan Huang, Jiankang Deng, Yun Ye, Junjie Huang, Xinze Chen,
                    Jiagang Zhu, Tian Yang, Jiwen Lu, Dalong Du, Jie Zhou},
                booktitle=  {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
                year=  {2021}
        }

    This dataset contains 2'059'906 identities and 42'474'558 images.


    .. warning::

      To use this dataset protocol, you need to have the original files of the WebFace42M dataset.
      Once you have it downloaded, please run the following command to set the path for Bob

        .. code-block:: sh

            bob config set bob.bio.face.webface42M.directory [WEBFACE42M PATH]

    """

    def __init__(
        self,
        database_path=rc.get("bob.bio.face.webface42M.directory", ""),
        transform=None,
    ):
        self.database_path = database_path

        if database_path == "":
            raise ValueError(
                "`database_path` is empty; please do `bob config set bob.bio.face.webface42M.directory` to set the absolute path of the data"
            )

        urls = WebFace42M.urls()
        filename = get_file(
            "webface42M.tar.gz",
            urls,
            file_hash="50c32cbe61de261466e1ea3af2721cea",
        )
        self.file = search_file(filename, "webface42M.csv")

        self._line_offset = 51
        self.transform = transform

    def __len__(self):
        # Avoiding this very slow task
        # return sum(1 for line in open(self.csv_file))
        return 42474558

    def __getitem__(self, idx):

        self.file.seek(0)

        # Allowing negative indexing
        if idx < 0:
            idx = self.__len__() + idx

        self.file.seek(idx * self._line_offset)
        line_sample = self.file.read(self._line_offset).split(",")

        label = int(line_sample[0])

        file_name = os.path.join(
            self.database_path, line_sample[1].rstrip("\n").strip()
        )
        image = bob.io.base.load(file_name)

        image = image if self.transform is None else self.transform(image)

        return {"data": image, "label": label}

    @staticmethod
    def urls():
        return [
            "https://www.idiap.ch/software/bob/databases/latest/webface42M.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/webface42M.tar.gz",
        ]

    def generate_csv(self, output_csv_directory):
        """
        Generates a bunch of CSV files containing all the files from the WebFace42M dataset
        The csv's have two columns only `LABEL, RELATIVE_FILE_PATH`



        Idiap file structure

          [0-9]_[0-6]_xxx
            |
            -- [0-9]_[0-9]_xxxxxxx
                |
                --- [0-9]_[0-9].jpg


        """
        label_checker = np.zeros(2059906)
        counter = 0

        # Navigating into the Idiap file structure
        for directory in os.listdir(self.database_path):
            output_csv_file = os.path.join(
                output_csv_directory, directory + ".csv"
            )

            with open(output_csv_file, "w") as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=",")

                print(f"Processing {directory}")
                rows = []
                path = os.path.join(self.database_path, directory)
                if not os.path.isdir(path):
                    continue
                for sub_directory in os.listdir(path):
                    sub_path = os.path.join(path, sub_directory)
                    label_checker[counter] = 1

                    if not os.path.isdir(sub_path):
                        continue

                    for file in os.listdir(sub_path):
                        relative_path = os.path.join(
                            directory, sub_directory, file
                        )
                        rows.append(
                            [
                                str(counter).zfill(7),
                                relative_path.rstrip("\n").rjust(42) + "\n",
                            ]
                        )
                        # csv_writer.writerow([label, relative_path])
                    counter += 1
                csv_writer.writerows(rows)

        # print(counter)
        # Checking if all labels were taken
        zero_labels = np.where(label_checker == 0)[0]
        if zero_labels.shape[0] > 0:
            print(zero_labels)
