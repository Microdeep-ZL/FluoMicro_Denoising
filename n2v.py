from PIL import Image
import numpy as np
from tensorflow.keras import layers


class N2VConfig:
    def __init__(self, file_paths,
                 ground_truth_paths=None,
                 patch_shape=(64, 64),
                 patches_per_batch=32,
                 epochs=20,
                 perc_pix=0.74,
                 data_augmentation=0,
                 conv="SeparableConv2D",
                 RGB=False,
                 validation_split=0
                 ):
        '''
        Parameter
        ---
        - file_paths: a list of tif file paths (noisy images)
        - ground_truth_paths: a list of tif file paths (clean images, not for training, only for evaluation)
            Note that `file_paths` and `ground_truth_paths` should correspond to each other
        - patch_shape: the shape of patch. Default (64, 64)
        - validation_split: the percentage of validation set from the whole dataset        
        - conv: Convolutional layer to be used in model. One of "Conv2D" and "SeparableConv2D". Defaults to 'SeparableConv2D'
        - RGB: Whether the image is RGB or gray. Defaults to False. 
        - data_augmentation: Integer from 0-7, represents how many additional folds you want the training images to have. 
            For example, 5 means the number of images will be multiplied by 6. Defaults to 0
        '''

        assert patch_shape[0] % 2 == 0 and patch_shape[1] % 2 == 0, 'Patch shape must be divisible by 2'
        if ground_truth_paths:
            assert len(file_paths)==len(ground_truth_paths), 'file_paths and ground_truth_paths must correspond to each other'
        self.file_paths = file_paths
        # todo 怎么用config设置predict和evaluate的参数
        self.ground_truth_paths = ground_truth_paths

        self.RGB = RGB
        channels = (3,) if RGB else (1,)
        self.input_shape = (None, None) + channels
        self.patch_shape = patch_shape
        # self.input_shape = patch_shape + channels
        self.patches_per_batch = patches_per_batch
        self.epochs = epochs
        self.perc_pix = perc_pix
        self.data_augmentation = data_augmentation
        self.Conv2D = layers.Conv2D if conv == "Conv2D" else layers.SeparableConv2D
        self.validation_split = validation_split

    def set_steps_per_epoch(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch

    def set_validation_steps(self, validation_steps):
        self.validation_steps = validation_steps


class N2VDataGenerator:
    '''
    Generate patches from multi-frame gray tiff files
    '''

    def __init__(self, config: N2VConfig):
        self.config = config
        self.prepare_is_validation()
        self.prepare_generator_steps()

    def prepare_is_validation(self):
        '''Prepare the dict {file_path: list[validation_frame]}'''
        self.is_validation = {}
        if self.config.validation_split:
            n_frames = 0  # Total number of frames in all tif files
            for file_path in self.config.file_paths:
                with Image.open(file_path) as image:
                    n_frames += image.n_frames

            validation_length = int(self.config.validation_split*n_frames)
            n_frames = np.arange(n_frames)
            np.random.shuffle(n_frames)
            validation_frames = n_frames[:validation_length]

            n_frames = 0
            for file_path in self.config.file_paths:
                temp = []
                with Image.open(file_path) as image:
                    for i in range(n_frames, n_frames+image.n_frames):
                        if i in validation_frames:
                            temp.append(i-n_frames)
                    n_frames += image.n_frames
                    self.is_validation[file_path] = temp

    def prepare_generator_steps(self):
        training_patches_total = 0
        validation_patches_total = 0
        patch_shape = self.config.patch_shape
        for file_path in self.config.file_paths:
            with Image.open(file_path) as image:
                a = len(range(0, image.size[0] -
                        patch_shape[0] + 1, patch_shape[0]))
                b = len(range(0, image.size[1] -
                        patch_shape[1] + 1, patch_shape[1]))
                validation_frames = len(self.is_validation.get(file_path, []))
                training_patches_total += a*b * \
                    (image.n_frames-validation_frames)
                validation_patches_total += a*b*validation_frames

        steps_per_epoch = int(training_patches_total /
                              self.config.patches_per_batch)
        validation_steps = int(validation_patches_total /
                               self.config.patches_per_batch)
        if self.config.data_augmentation:
            validation_steps *= self.config.data_augmentation + 1
            steps_per_epoch *= self.config.data_augmentation + 1
        self.config.set_steps_per_epoch(steps_per_epoch)
        self.config.set_validation_steps(validation_steps)

    def get_validation_batch(self):
        if self.config.validation_split:  # Otherwise, return None
            return self.get_training_batch(validation=True)

    def get_training_batch(self, validation=False):
        '''
        Return (patches, targets)
            patches: (patches_per_batch, height, width, channels)
            targets: (patches_per_batch, height, width, channels+1)
        '''
        patch_target_generator = self._get_patch_target(validation)
        while 1:
        # try:
            patches, targets = [], []
            for _ in range(self.config.patches_per_batch):
                patch, target = next(patch_target_generator)
                patches.append(patch)
                targets.append(target)
        # except RuntimeError:
        #     break
        # finally:
        # if patches:
            patches = np.concatenate(patches)
            targets = np.concatenate(targets)
            yield patches, targets

    def get_evaluation_data(self, batch_size=1):
        '''
        Return (noisy_images, clean_images)
            noisy_images: (batch_size, height, width, channels)
            clean_images: (batch_size, height, width, channels)
        '''

        noisy_clean_generator = self._get_noisy_clean()
        while 1:
            try:
                noisy_images, clean_images = [], []
                for _ in range(batch_size):
                    noisy, clean = next(noisy_clean_generator)
                    if noisy is None:
                        yield None, None
                        break
                    noisy_images.append(noisy)
                    clean_images.append(clean)   
            except StopIteration:
                break
            # except RuntimeError:
            #     yield None, None
            finally:
                if noisy_images:
                    noisy_images = np.concatenate(noisy_images)
                    clean_images = np.concatenate(clean_images)
                    yield noisy_images, clean_images

    def _get_noisy_clean(self):
        '''
        Return (noisy_image, clean_image), float32, already normalized to 01 interval
            noisy_image: (1, height, width, channels)
            clean_image: (1, height, width, channels)
        '''
        noisy_paths=self.config.file_paths
        clean_paths=self.config.ground_truth_paths
        images_total=0
        for i in range(len(noisy_paths)):
            with Image.open(noisy_paths[i]) as noisy_image:
                with Image.open(clean_paths[i]) as clean_image:
                    images_total+=noisy_image.n_frames
                    assert noisy_image.n_frames==clean_image.n_frames, "The number of noisy and clean images must be equal"
        counter=0
        for i in range(len(noisy_paths)):
            with Image.open(noisy_paths[i]) as noisy_image:
                with Image.open(clean_paths[i]) as clean_image:
                    for j in range(noisy_image.n_frames):
                            noisy_image.seek(j)
                            clean_image.seek(j)                        
                            noisy = np.array(noisy_image).astype("float32")
                            clean = np.array(clean_image).astype("float32")

                            axis = 0 if self.config.RGB else (0,-1)
                            noisy = np.expand_dims(noisy, axis)
                            clean = np.expand_dims(clean, axis)

                            noisy = self._normalization(noisy)
                            clean = self._normalization(clean)
                            yield noisy, clean
                            counter+=1
                            print(f"{counter}/{images_total} images have been processed", end="\r")
            yield None, None
        print()

    def _load_images(self, validation=False):
        '''Return image array (height, width) or (height, width, channels), depending on the image.'''
        while 1:
            for file_path in self.config.file_paths:
                with Image.open(file_path) as image:
                    # 随机打乱图片的顺序
                    for i in self._shuffle_range(image.n_frames):
                        if validation == (i in self.is_validation.get(file_path, [])):
                            image.seek(i)
                            # yield np.array(image)
                            yield np.array(image).astype("float32")

                            if self.config.data_augmentation:
                                for j in self._shuffle_range(7)[:self.config.data_augmentation]:
                                    transposed_image = image.transpose(j)
                                    # yield np.array(transposed_image)
                                    yield np.array(transposed_image).astype("float32")

    def _normalization(self, image):
        '''Return the image array normalized to 01 interval'''
        m = image.max()
        n = image.min()
        image = (image-n)/(m-n)
        return image

    def _get_patch_target(self, validation=False):
        '''
        Return (patch, target)
            patch: (1, height, width, channels)
            target: (1, height, width, channels+1)
        '''
        image_generator = self._load_images(validation)
        while 1:
            image = next(image_generator)
            image = self._normalization(image)

            image_shape = image.shape
            patch_shape = self.config.patch_shape

            assert image_shape[0] >= patch_shape[0] and image_shape[1] >= patch_shape[1], "Patch shape cannot be larger than image shape"

            # 如果尺寸不能整除，就补镜像
            if image_shape[0] % patch_shape[0] != 0 or image_shape[1] % patch_shape[1] != 0:
                image = self._padding_image(image)
            # 随机打乱patch的顺序
            for i in self._shuffle_range(0, image_shape[0] - patch_shape[0] + 1, patch_shape[0]):
                for j in self._shuffle_range(0, image_shape[1] - patch_shape[1] + 1, patch_shape[1]):
                    if image.ndim == 2:
                        patch = image[np.newaxis,
                                      i:i+patch_shape[0],
                                      j:j+patch_shape[1],
                                      np.newaxis]
                    else:
                        patch = image[np.newaxis,
                                      i:i+patch_shape[0],
                                      j:j+patch_shape[1]
                                      ]

                    target = self._manipulate(patch)
                    yield patch, target

    def _padding_image(self, image):
        '''
        Parameter
        ---
        image: (height, width) or (height, width, channels)
        '''
        b = np.flip(image, 1)  # 右
        c = np.flip(image, 0)  # 下
        d = np.flip(image, (0, 1))  # 右下

        b = np.concatenate((image, b), 1)
        d = np.concatenate((c, d), 1)
        image = np.concatenate((b, d))
        return image

    def _shuffle_range(self, *args):
        '''Similar to `range()`, but the order is shuffled'''
        shuffled_list = list(range(*args))
        np.random.shuffle(shuffled_list)
        return shuffled_list

    def _manipulate(self, patch):
        '''Manipulate values in a patch, and return the target

        Parameter
        ---
        patch: (1, height, width, channels)

        Return
        ---
        target: (1, height, width, channels+1)
        '''
        coords = self._get_stratified_coords()

        if self.config.RGB:
            indexing = (0,) + coords + (slice(3),)
            indexing_mask = (0,) + coords + (3,)
        else:
            indexing = (0,) + coords + (slice(1),)
            indexing_mask = (0,) + coords + (1,)

        target = np.concatenate(
            [patch, np.zeros_like(patch[..., 0, np.newaxis])], axis=-1)
        target[indexing_mask] = 1
        patch[indexing] = self._value_manipulation(patch[0], coords)
        return target

    def _get_stratified_coords(self):
        box_size = np.sqrt(100 / self.config.perc_pix)
        shape = self.config.patch_shape

        box_count_y = int(shape[0] / box_size)
        box_count_x = int(shape[1] / box_size)
        x_coords = []
        y_coords = []
        for i in range(box_count_y):
            for j in range(box_count_x):
                y = np.random.rand() * box_size
                x = np.random.rand() * box_size
                y = int(i * box_size + y)
                x = int(j * box_size + x)
                if (y < shape[0] and x < shape[1]):
                    y_coords.append(y)
                    x_coords.append(x)
        return y_coords, x_coords

    def _value_manipulation(self, patch, coords):
        '''
        Return nearby pixel values to replace the manipulated pixels.

        Parameters
        ---
        patch           : ndarray(height, width, channels)
        coords          : tuple(y_coords, x_coords)
        subpatch_radius : int, use surrounding pixel to raplace the center pixels of subpatche
        '''
        vals = []
        for coord in zip(*coords):
            sub_patch = self._get_subpatch(patch, coord)
            rand_coords = tuple(np.random.randint(s)
                                for s in sub_patch.shape[:2])
            vals.append(sub_patch[rand_coords])
        return vals

    def _get_subpatch(self, patch, coord, subpatch_radius=5):
        '''
        Get a square subpatch of size (2*subpatch_radius, 2*subpatch_radius)

        Parameters
        -----
        patch   : (height, width, channels)
        coord   : (y, x)

        Return
        ---
        subpatch: (sub_height, sub_width, channels)
        '''
        start = np.maximum(0, np.array(coord) - subpatch_radius)
        end = start + subpatch_radius * 2 + 1

        shift = np.minimum(0, patch.shape[:2] - end)

        start += shift
        end += shift

        slices = tuple(slice(s, e) for s, e in zip(start, end))

        return patch[slices]
