from tensorflow.keras import layers, models, Model, Input, callbacks, optimizers
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import time
import numpy as np
from skimage import io
from math import ceil
import os


class Rescaling(layers.Layer):
    '''
    Rescale the image into 01 interval.
    The built-in `tf.keras.layers.Rescaling()` couldn't work here, because it only receives fixed float arguments.
    What we want is to rescale each image according to its maximum and minimum pixel values, rather than any fixed values.
    '''

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        m = tf.reduce_max(inputs)
        n = tf.reduce_min(inputs)
        scale = 1/(m-n)
        offset = -n/(m-n)
        return inputs * scale + offset


class Unet:
    '''
    Parameter
    ---
    model_summary: Bool. Defaults to False. Whether to print model summary and plot the topograhpy.'''

    def __init__(self, config, model_summary=False):
        self.config = config
        self.model = self.define_model()
        if model_summary:
            self.model.summary()
            plot_model(self.model, type(self).__name__+".png", show_shapes=True)

        self.compiled = False

    def define_model(self, n_depth=2):
        '''
        Assume the image has shape (height, width, channels)
        Parameters
        ---
        n_depth:    Defaults to 2, it's suitable for image of size about (128, 128). the depth of contracting path, i.e. the number of MaxPooling2D layers in total. 
                    If n_depth is too big, image of small size will not be properly processed
        '''
        inputs = Input(shape=self.config.input_shape)
        # todo 当图块尺寸为奇数时，concatenate层报错
        # inputs = Input(shape=(567,567,1))

        # 不能用于输入图块，容易导致黑色背景变得很白
        # inputs = Rescaling()(inputs)

        contracting = []
        for i in range(n_depth+1):
            block = models.Sequential(name='contracting_'+str(i))
            if i != 0:
                block.add(layers.MaxPooling2D())

            # 第一个卷积层含64个filters，最后一个卷积层含256个filters
            block.add(self.config.Conv2D(
                2**(i+6), 3, use_bias=False, padding='same'))
            block.add(layers.BatchNormalization())
            block.add(layers.Activation("relu"))

            block.add(self.config.Conv2D(
                2**(i+6), 3, use_bias=False, padding='same'))
            block.add(layers.BatchNormalization())
            block.add(layers.Activation("relu"))
            contracting.append(block)

        expanding = []
        for i in range(n_depth):
            block = models.Sequential(name='expanding_'+str(i))
            if i != 0:
                # 唯一的卷积层含128个filters
                block.add(self.config.Conv2D(2**(6+n_depth-i),
                          3, use_bias=False, padding='same'))
                block.add(layers.BatchNormalization())
                block.add(layers.Activation("relu"))

                block.add(self.config.Conv2D(2**(6+n_depth-i),
                          3, use_bias=False, padding='same'))
                block.add(layers.BatchNormalization())
                block.add(layers.Activation("relu"))

            block.add(layers.Conv2DTranspose(2**(6+n_depth-i), 2, strides=2))
            expanding.append(block)

        ending = models.Sequential([
            self.config.Conv2D(2**6, 3, use_bias=False, padding='same'),
            layers.BatchNormalization(),
            layers.Activation("relu"),

            self.config.Conv2D(2**6, 3, use_bias=False, padding='same'),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            # todo 最终的单点预测输出
            # todo 尺寸和原图相等, 已经做到了，但是和n2v的方法一样吗？
            # todo Noise2Void到底是怎么做的，还是得看源码和论文
            layers.Conv2D(1, 1),

            # Normalize to 01 interval
            Rescaling()

        ], name='ending')

        for i in range(len(contracting)):
            if i == 0:
                contracting[i] = contracting[i](inputs)
            else:
                contracting[i] = contracting[i](contracting[i-1])

        for i in range(len(expanding)):
            if i == 0:
                expanding[i] = expanding[i](contracting[-1])
            else:
                expanding[i] = expanding[i](expanding[i-1])

            
            # _, height_contracting, width_contracting, _ = contracting[-i-2].shape
            # _, height_expanding, width_expanding, _ = expanding[i].shape
            # height_crop = (height_contracting-height_expanding)//2
            # width_crop = (width_contracting-width_expanding)//2
            # contracting_cropped = layers.Cropping2D(
            #     cropping=(height_crop, width_crop))(contracting[-i-2])

            # expanding[i] = layers.Concatenate()(
            #     [expanding[i], contracting_cropped])

            expanding[i] = layers.Concatenate()(
                [expanding[i], contracting[-i-2]])

        outputs = ending(expanding[-1])
        outputs=layers.add([outputs,inputs]) # Residual connection

        return Model(inputs, outputs)

    def train(self, data_generator, early_stopping_patience=10, reduce_lr_patience=5, reduce_lr_factor=0.7):
        '''
        Parameter
        -
        - data_generator: N2VDataGenerator instance
        - early_stopping_patience: argument for callback EarlyStopping
        - reduce_lr_patience: argument for callback ReduceLROnPlateau
        - reduce_lr_factor: argument for callback ReduceLROnPlateau
        '''
        if not self.compiled:
            self.compile()
        # x, y=next(data_generator)
        # return self.model.fit(x,y,batch_size=32,epochs=1)

        callbacks_list = [callbacks.TensorBoard(log_dir="tensorboard"),
                          callbacks.EarlyStopping(
                              monitor="val_loss", patience=early_stopping_patience),
                          callbacks.ModelCheckpoint(
                              filepath="ckpt/best", monitor="val_loss", save_best_only=True, save_weights_only=True),
                          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=0.00005)]
        print("TRAINING BEGINS".center(40, '-'))
        return self.model.fit(data_generator.get_training_batch(),
                              validation_data=data_generator.get_validation_batch(),
                              callbacks=callbacks_list,
                              steps_per_epoch=self.config.steps_per_epoch,
                              validation_steps=self.config.validation_steps,
                              epochs=self.config.epochs)

        #   #   todo Remove, just to test
        #   steps_per_epoch=10,
        #   validation_steps=6,
        #   epochs=4)

    def compile(self):
        # todo 选择更优的optimizer
        # 选择更优的参数 
        # 训练过程不计算PSNR和SSIM
        optimizer=optimizers.RMSprop(momentum=0.1)
        self.model.compile(optimizer, loss=self.loss)
        self.compiled = True

    @tf.function
    def psnr(self, y_true, y_pred):
        '''
        Parameter
        ---
        y_true: (batch_size, height, width, channels), float32, 01 interval, there's no additional channel as mask
        y_pred: (batch_size, height, width, channels), float32, 01 interval
        '''
        # m = tf.reduce_max(y_true)
        # n = tf.reduce_min(y_true)
        # y_true = (y_true-n)/(m-n)
        # print("psnr:", tf.executing_eagerly())
        return tf.image.psnr(y_true, y_pred, 1)

    @tf.function
    def ssim(self, y_true, y_pred):
        # print("ssim:", tf.executing_eagerly())
        return tf.image.ssim(y_true, y_pred, 1)

    def loss(self, y_true, y_pred):
        '''
        Parameter
        ---
        y_true: (batch_size, height, width, channels+1), float32, 01 interval, the additional channel is mask
        y_pred: (batch_size, height, width, channels), float32, 01 interval
        '''
        coords = y_true[..., -1] == 1
        # 归一化
        # y_true = tf.cast(y_true[..., :-1], tf.float32)
        y_true = y_true[..., :-1]

        m = tf.reduce_max(y_true)
        n = tf.reduce_min(y_true)
        y_true = (y_true-n)/(m-n)

        squared_difference = tf.reduce_mean(
            tf.square(y_true[coords] - y_pred[coords]), axis=-1)
        return squared_difference

    def predict(self, data_generator, save_dir, divide=1, batch_size = 1):
        '''
        The denoised image will be saved in tif file type and uint8 format.

        Parameter
        ---
        - data_generator: Pass the `file_paths` and `RGB` arguments to N2VDataGenerator.
        - save_dir: see doc for `evaluate()` 
        - divide: see doc for `evaluate()` 
        - batch_size: see doc for `evaluate()` 
        '''
        print("PREDICTION BEGINS".center(40, '-'))
        restored = []
        counter, duration = 0, 0
        file_number = 0
        for x in data_generator.get_prediction_data(batch_size):
            if x is None:
                self._save_images(restored, save_dir, file_number)
                restored = []
                file_number += 1
                continue
            counter += 1
            begin = time.time()

            # 可能OOM，则需要拆分处理，再拼合图像。拼合时需要考虑边界
            if divide == 1:
                restored_image = self.model.predict_on_batch(x)
            else:
                restored_image = np.zeros_like(x)
                # 如果不能除尽，则要向上取整，保证每个像素点都被取到
                i_step = ceil(x.shape[1]/divide)
                j_step = ceil(x.shape[2]/divide)
                for i in range(divide):
                    for j in range(divide):
                        restored_image[:, i*i_step:(i+1)*i_step, j*j_step:(j+1)*j_step, :] = self.model.predict_on_batch(
                            x[:, i*i_step:(i+1)*i_step, j*j_step:(j+1)*j_step, :])

            duration += time.time()-begin


            restored.append(restored_image)

        duration = np.round(duration/counter/batch_size, 1)
        print(f"{duration}s on average to process each image")

    def evaluate(self, data_generator, save_dir=None, divide=1, batch_size = 1):
        '''
        Parameter
        ---
        - data_generator: Pass the `file_paths` and `ground_truth_paths` arguments to N2VDataGenerator.
        - divide: Integer, defaults to 1. When predicting or evaluating, large images might lead to OOM (out of memory).
            In that case, you can divide a large image into multiple smaller patches. 
            For example, 2 means the model will process 4 smaller images, and then combine them together.
        - save_dir: String, path to save the restored result. The model won't save restored images by default.
            For now, only support tif file type and uint8 format.
        - batch_size: Integer, the number of images to be processed in each batch. Defaults to 1

        Return
        --- 
        Dict whose keys include `duration`, `ssim`, `psnr`, `old_ssim`, `old_psnr`
            - duration: how many seconds it needs to process one image
            - ssim, psnr: between restored image and ground truth
            - old_ssim, old_psnr: between noisy image and ground truth
        '''
        print("EVALUATION BEGINS".center(40, '-'))
        SSIM, PSNR, OLD_SSIM, OLD_PSNR = [], [], [], []
        restored = []
        counter, duration = 0, 0
        
        file_number = 0
        for x, y in data_generator.get_evaluation_data(batch_size):
            if x is None:
                self._save_images(restored, save_dir, file_number)
                restored = []
                file_number += 1
                continue
            counter += 1
            begin = time.time()

            # 可能OOM，则需要拆分处理，再拼合图像。拼合时需要考虑边界
            if divide == 1:
                restored_image = self.model.predict_on_batch(x)
            else:
                restored_image = np.zeros_like(x)
                # 如果不能除尽，则要向上取整，保证每个像素点都被取到
                i_step = ceil(x.shape[1]/divide)
                j_step = ceil(x.shape[2]/divide)
                for i in range(divide):
                    for j in range(divide):
                        restored_image[:, i*i_step:(i+1)*i_step, j*j_step:(j+1)*j_step, :] = self.model.predict_on_batch(
                            x[:, i*i_step:(i+1)*i_step, j*j_step:(j+1)*j_step, :])

            duration += time.time()-begin

            old_ssim, old_psnr = self.ssim(y, x), self.psnr(y, x)
            new_ssim, new_psnr = self.ssim(
                y, restored_image), self.psnr(y, restored_image)

            SSIM.append(new_ssim)
            PSNR.append(new_psnr)
            OLD_SSIM.append(old_ssim)
            OLD_PSNR.append(old_psnr)
            if save_dir:
                restored.append(restored_image)

        # SSIM = np.mean(SSIM).round(2)
        # PSNR = np.mean(PSNR).round(2)
        # OLD_SSIM = np.mean(OLD_SSIM).round(2)
        # OLD_PSNR = np.mean(OLD_PSNR).round(2)
        duration = np.round(duration/counter/batch_size, 1)
        result = {"duration": duration, "ssim": SSIM, "psnr": PSNR,
                  "old_ssim": OLD_SSIM, "old_psnr": OLD_PSNR}

        return result

    def _save_images(self, restored, save_dir, file_number):
        # 输入文件有多少个，输出文件就应该有多少个
        if save_dir:
            restored = np.concatenate(restored)
            # 保存为uint8
            # todo 保存100张图片时，也可能会OOM
            # 可以先单张转为uint8，再concatenate尝试解决
            restored = np.around(restored*255).astype("uint8")
            try:
                # 如果是灰度图像，去掉channel轴
                restored = np.squeeze(restored, axis=-1)
            except ValueError:
                pass

            if not os.path.isdir(save_dir): os.makedirs(save_dir)
            file_name = f"restored_{file_number}.tif"
            save_path = os.path.join(save_dir, file_name)
            io.imsave(save_path, restored)