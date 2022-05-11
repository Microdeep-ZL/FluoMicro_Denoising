from PIL import Image
import numpy as np


class N2VConfig:
    def __init__(self, file_paths,
                 ground_truth_paths=None,
                 patch_shape=(64, 64),
                 patches_per_batch=32,
                 perc_pix=1,
                 data_augmentation=True,
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
        - perc_pix: Float. Percentage of pixels to be manipulated. 
            eg. 0.25 means an image of size (100, 100) will have 0.25% ie. 25 pixels manipulated.
            Or equivently every subpatch of size (20, 20) has a pixel manipulated where 20=sqrt(1/0.0025)
        - validation_split: the percentage of validation set from the whole dataset        
        - RGB: Whether the image is RGB or gray. Defaults to False. 
        - data_augmentation: Bool, True by default. If true, the image will be randomly flipped or rotated.
        '''

        assert patch_shape[0] % 2 == 0 and patch_shape[1] % 2 == 0, 'Patch shape must be divisible by 2'
        if ground_truth_paths:
            assert len(file_paths)==len(ground_truth_paths), 'file_paths and ground_truth_paths must correspond to each other'
        self.file_paths = file_paths
        self.ground_truth_paths = ground_truth_paths

        self.RGB = RGB
        self.patch_shape = patch_shape
        self.patches_per_batch = patches_per_batch
        self.perc_pix = perc_pix
        self.data_augmentation = data_augmentation
        self.validation_split = validation_split


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
        # if self.config.data_augmentation:
        #     validation_steps *= self.config.data_augmentation + 1
        #     steps_per_epoch *= self.config.data_augmentation + 1
        self.steps_per_epoch=steps_per_epoch
        self.validation_steps=validation_steps

    def get_validation_batch(self, supervised=False):
        if self.config.validation_split:  # Otherwise, return None
            return self.get_training_batch(validation=True, supervised=supervised)

    def get_training_batch(self, validation=False, supervised=False):
        '''
        Return (patches, targets)
            patches: (patches_per_batch, height, width, channels)
            targets: (patches_per_batch, height, width, channels+1)
        '''
        patch_target_generator = self._get_patch_target(validation, supervised)
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

    def get_prediction_data(self, batch_size=1):
        '''
        Return noisy_images: (batch_size, height, width, channels)
        '''
        # 逻辑基本上和get_evaluation_data()一样，但是本方法写得更简单
        def _get_image():
            counter,images_total=0,0
            for file_path in self.config.file_paths:
                with Image.open(file_path) as image:
                    images_total+=image.n_frames
            for file_path in self.config.file_paths:
                with Image.open(file_path) as image:
                    for i in range(image.n_frames):
                        image.seek(i)                    
                        noisy= np.array(image).astype("float32")
                        axis = 0 if self.config.RGB else (0,-1)
                        noisy = np.expand_dims(noisy, axis)
                        noisy = self._normalization(noisy)

                        yield noisy
                        counter+=1
                        print(f"{counter}/{images_total} images have been processed",end="\r")
                yield None # 提示一个tif文件已处理完
            print()

        images=[]
        for image in _get_image():
            if image is None:
                if images:
                    yield np.concatenate(images)
                    images=[]
                yield image
            else:
                images.append(image)
                if len(images)==batch_size:
                    yield np.concatenate(images)
                    images=[]
                
    def get_evaluation_data(self, batch_size):
        '''
        Return (noisy_images, clean_images)
            noisy_images: (batch_size, height, width, channels)
            clean_images: (batch_size, height, width, channels)
        '''
        noisy_images, clean_images = [], []
        for noisy, clean in self._get_noisy_clean():
            if noisy is None:
                if noisy_images:
                    noisy_images = np.concatenate(noisy_images)
                    clean_images = np.concatenate(clean_images)
                    yield noisy_images, clean_images
                    noisy_images, clean_images = [], []
                yield None, clean
            else:
                noisy_images.append(noisy)
                clean_images.append(clean)
                if len(noisy_images)==batch_size:
                    noisy_images = np.concatenate(noisy_images)
                    clean_images = np.concatenate(clean_images)
                    yield noisy_images, clean_images                
                    noisy_images, clean_images = [], []

    def _get_noisy_clean(self):
        '''
        Return (noisy_image, clean_image), float32, already normalized to 01 interval
            noisy_image: (1, height, width, channels)
            clean_image: (1, height, width, channels)
        '''
        assert self.config.ground_truth_paths is not None, "You have to pass the `ground_truth_paths` argument to N2VDataGenerator"
        noisy_paths=self.config.file_paths
        clean_paths=self.config.ground_truth_paths
        images_total=0
        for i in range(len(noisy_paths)):
            with Image.open(noisy_paths[i]) as noisy_image, Image.open(clean_paths[i]) as clean_image:
                images_total+=noisy_image.n_frames
                assert noisy_image.n_frames==clean_image.n_frames, "The number of noisy and clean images must be equal"
        counter=0
        for i in range(len(noisy_paths)):
            with Image.open(noisy_paths[i]) as noisy_image, Image.open(clean_paths[i]) as clean_image:
                for j in range(noisy_image.n_frames):
                    noisy_image.seek(j);clean_image.seek(j)                        
                    noisy = np.array(noisy_image, dtype="float32")
                    clean = np.array(clean_image, dtype="float32")

                    axis = 0 if self.config.RGB else (0,-1)
                    noisy = np.expand_dims(noisy, axis)
                    clean = np.expand_dims(clean, axis)

                    noisy = self._normalization(noisy)
                    clean = self._normalization(clean)
                    yield noisy, clean
                    counter+=1
                    print(f"{counter}/{images_total} images have been processed", end="\r")
            yield None, noisy_paths[i] #第2项提示noisy image的文件名
        print()

    def _load_images(self, validation, supervised):
        '''
        Return (image_array, None) or (image_array, ground_truth_array)
        image array (height, width) or (height, width, channels), depending on the image.
        '''
        while 1:
            for j in range(len(self.config.file_paths)):
                file_path=self.config.file_paths[j]
                if supervised:
                    gt_path=self.config.ground_truth_paths[j]
                    gt=Image.open(gt_path)
                with Image.open(file_path) as image:
                    # 随机打乱图片的顺序
                    for i in self._shuffle_range(image.n_frames):
                        if validation == (i in self.is_validation.get(file_path, [])):
                            image.seek(i)
                            if supervised: gt.seek(i)
                            # yield np.array(image)
                            if self.config.data_augmentation and (not validation) and np.random.randint(8)!=0:
                                # 八分之一概率不做数据增强
                                aug=self._shuffle_range(7)[0]
                                transposed_image = image.transpose(aug)
                                yield (self._image_to_array(transposed_image), self._image_to_array(gt.transpose(aug)) if supervised else None)
                            else:
                                yield (self._image_to_array(image), self._image_to_array(gt) if supervised else None)
                    if supervised: gt.close()

    def _image_to_array(self,image):
        array=np.array(image, dtype="float32")
        # Uniform distribution [0.00002, 0.0001) and [0.00005, 0.0004)
        # percent_left=np.random.ranf()*(0.0001-0.00002)+0.00002
        # percent_right=np.random.ranf()*(0.0004-0.00005)+0.00005
        # array=self._normalization(array,percent_left,percent_right)
        array=self._normalization(array)
        return array

    def _normalization(self, image, percent_left=0.1**4, percent_right=0.1**7):
        '''Return the image array normalized to 01 interval'''
        histogram=np.sort(image.flatten())
        n=histogram[int(percent_left*len(histogram))]
        m=histogram[-1-int(percent_right*len(histogram))]
        image=np.clip(image,n,m)
        return (image-n)/(m-n)

    def _get_patch_target(self, validation, supervised):
        '''
        Return (patch, target)
            patch: (1, height, width, channels)
            target: (1, height, width, channels+1)
        '''
        image_generator = self._load_images(validation, supervised)
        while 1:
            image, gt = next(image_generator)
            # image = self._normalization(image)

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
                    # return patch
                    if supervised:
                        if image.ndim==2:
                            target = gt[np.newaxis,
                                      i:i+patch_shape[0],
                                      j:j+patch_shape[1],
                                      np.newaxis]
                        else:
                            target = gt[np.newaxis,
                                        i:i+patch_shape[0],
                                        j:j+patch_shape[1]
                                        ] 
                    else:
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


class DataGenerator:
    def __init__(self, file_path, gt_path, batch_size=1, validation_split=0):
        self.file_path=file_path
        self.gt_path=gt_path
        self.batch_size=batch_size
        self.validation_split=validation_split
        with Image.open(file_path) as noisy:
            if validation_split:
                li=np.arange(noisy.n_frames)            
                np.random.shuffle(li)
                a=int(validation_split*noisy.n_frames)
                self.validation_list=li[:a]
                self.steps_per_epoch=(noisy.n_frames-a)*4//batch_size
                self.validation_steps=a//batch_size
            else:
                self.validation_list=[]
                self.validation_steps=0
                self.steps_per_epoch=noisy.n_frames*4//batch_size

    def get_training_batch(self, validation=False):
        image_generator=self.get_noisy_clean(validation)
        # while 1:
        #     noisy, gt=[],[]
        #     for i in range(self.batch_size):
        #         a,b=next(image_generator)
        #         noisy.append(a)
        #         gt.append(b)
        #     yield np.concatenate(noisy), np.concatenate(gt)
        while 1:
            noisy, gt=[],[]
            a,b=next(image_generator)
            for i in range(self.batch_size):
                noisy.append(a)
                gt.append(b)
            yield np.concatenate(noisy), np.concatenate(gt)

    def get_validation_batch(self):
        if self.validation_split:  # Otherwise, return None
            return self.get_training_batch(validation=True)

    def get_noisy_clean(self, validation=False):
        with Image.open(self.file_path) as noisy, Image.open(self.gt_path) as gt:
            assert noisy.n_frames==gt.n_frames,"The number of noisy and clean images are not equal"
            while 1:
                for i in range(noisy.n_frames):
                    noisy.seek(i)
                    gt.seek(i)
                    if validation==(i in self.validation_list):
                        num=np.random.randint(8)
                        # 数据增强
                        if num!=7:
                            transposed_noisy=noisy.transpose(num)
                            transposed_gt=gt.transpose(num)
                            noisy_arr=np.array(transposed_noisy, dtype="float32")
                            gt_arr=np.array(transposed_gt, dtype="float32")
                        else:
                            noisy_arr=np.array(noisy, dtype="float32")
                            gt_arr=np.array(gt, dtype="float32")
                        noisy_arr,gt_arr=self._normalization(noisy_arr), self._normalization(gt_arr)
                        for j in range(2):
                            for k in range(2):
                                a=noisy_arr[np.newaxis, j*1024:(j+1)*1024,k*1024:(k+1)*1024 , np.newaxis]
                                b=gt_arr[np.newaxis, j*1024:(j+1)*1024,k*1024:(k+1)*1024 , np.newaxis]
                                yield a,b

    def _normalization(self, image, percent_left=0.1**4, percent_right=0.1**7):
        '''Return the image array normalized to 01 interval'''
        histogram=np.sort(image.flatten())
        n=histogram[int(percent_left*len(histogram))]
        m=histogram[-int(percent_right*len(histogram))]
        image=np.clip(image,n,m)
        return (image-n)/(m-n)
