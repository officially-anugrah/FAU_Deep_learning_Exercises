import numpy as np
import matplotlib.pyplot as plt
import json
import random
from skimage.transform import resize

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        with open(label_path, 'r') as f:
            self.labels = json.load(f)

        self.image_list = list(self.labels.keys())

        self.indices = list(range(len(self.image_list)))
        self.epoch = 0
        self.current_index = 0

        if self.shuffle:
            random.shuffle(self.indices)

    def next(self):
        batch_images = []
        batch_labels = []

        for _ in range(self.batch_size):
            if self.current_index >= len(self.image_list):
                self.epoch += 1
                self.current_index = 0
                if self.shuffle:
                    random.shuffle(self.indices)

            image_filename = self.image_list[self.indices[self.current_index]]
            label = self.labels[image_filename]

            image = np.load(f"{self.file_path}/{image_filename}.npy")
            image = resize(image, self.image_size)

            if self.rotation:
                image = np.rot90(image, random.choice([0, 1, 2, 3]))

            if self.mirroring and random.random() > 0.5:
                image = np.fliplr(image)

            batch_images.append(image)
            batch_labels.append(label)

            self.current_index += 1

        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)

        return batch_images, batch_labels

    def current_epoch(self):
        return self.epoch

    def class_name(self, label):
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return classes[label]

    def show(self):
        images, labels = self.next()

        plt.figure(figsize=(12, 12))
        for i in range(len(images)):
            ax = plt.subplot(4, 3, i + 1)
            ax.imshow(images[i], cmap='gray')
            ax.set_title(self.class_name(labels[i]))
            ax.axis('off')

        plt.show()

image_gen = ImageGenerator(
    file_path = './exercise_data/',
    label_path = './Labels.json',
    batch_size = 9,
    image_size = [128, 128, 3],
    rotation = True,
    mirroring = True,
    shuffle = True
)

image_gen.show()