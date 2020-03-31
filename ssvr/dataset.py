import random

import numpy as np

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from ssvr.utils import batched, files

# TODO: instead of rotating each image once create all
#       possible rotations for each image (dataset x4)
#       but then think hard about train vs val split

class RotatedDataset(Sequence):
    angles = [0, 90, 180, 270]

    def __init__(self, root, target_size, preprocessing_function=None, data_format=None, shuffle=False, batch_size=1):
        super().__init__()

        self.target_size = target_size
        self.preprocessing_function = preprocessing_function
        self.data_format = data_format
        self.shuffle = shuffle

        # paths[i] contains all file paths for batch i
        # note: possibile that last batch size < batch_size
        self.paths = list(batched(files(root), batch_size))

        # generate random rotations for all batches once
        self.labels = [[random.randrange(len(self.angles)) for _ in range(len(batch))]
                       for batch in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        paths = self.paths[i]
        labels = self.labels[i]

        inputs = []
        targets = []

        for path, label in zip(paths, labels):
            image = load_img(path, target_size=self.target_size, interpolation="bilinear")
            image = image.rotate(self.angles[label])

            array = img_to_array(image, data_format=self.data_format)

            if self.preprocessing_function is not None:
                array = self.preprocessing_function(array)

            inputs.append(array)
            targets.append(label)

        return np.array(inputs), np.array(targets)

    def on_epoch_end(self):
        if self.shuffle:
            batch = list(zip(self.paths, self.labels))
            random.shuffle(batch)
            self.paths, self.labels = list(zip(*batch))


# These clean and nice dataset abstractions below heavily inspired by pytorch

class Subset(Sequence):
    def __init__(self, dataset, indices):
        super().__init__()

        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

    def on_epoch_end(self):
        self.dataset.on_epoch_end()


def random_split(dataset, lengths):
    assert sum(lengths) == len(dataset)

    indices = list(range(len(dataset)))

    random.shuffle(indices)

    return [Subset(dataset, indices[offset - length : offset])
            for (offset, length) in zip(np.cumsum(lengths).tolist(), lengths)]
