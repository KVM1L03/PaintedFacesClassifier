import os
import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

ia.seed(1)

# Define the augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))), # gaussian blur
    iaa.LinearContrast((0.75, 1.5)), # contrast adjustment
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # noise
    iaa.Multiply((0.8, 1.2), per_channel=0.2), # brightness adjustment
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, 
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, 
                rotate=(-25, 25), 
                shear=(-8, 8),
                mode='edge') 
], random_order=True)

# Define the directories
directories = ['Dataset/angry_face', 'Dataset/happy_face', 'Dataset/sad_face']

for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith(".png"): 
            image_path = os.path.join(directory, filename)
            image = imageio.imread(image_path)
            images = np.array([image for _ in range(100)], dtype=np.uint8)  # Changed from 50 to 100
            images_aug = seq(images=images)
            for i, image_aug in enumerate(images_aug):
                imageio.imsave(f'{directory}/{filename[:-4]}_aug{i}.png', image_aug)