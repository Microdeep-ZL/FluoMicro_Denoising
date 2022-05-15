from tensorflow.keras import layers, models, Model, Input, callbacks, optimizers
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import time
import numpy as np
from skimage import io
from math import ceil
import os


# class Rescaling(layers.Layer):
#     '''
#     Rescale the image into 01 interval.
#     The built-in `tf.keras.layers.Rescaling()` couldn't work here, because it only receives fixed float arguments.
#     What we want is to rescale each image according to its maximum and minimum pixel values, rather than any fixed values.
#     '''

#     def call(self, inputs):
#         inputs = tf.cast(inputs, dtype=tf.float32)
#         m = tf.reduce_max(inputs)
#         n = tf.reduce_min(inputs)
#         scale = 1/(m-n)
#         offset = -n/(m-n)
#         return inputs * scale + offset


class Unet:
    def __init__(self, n_depth=3, kernel_size=5, RGB=False, conv="SeparableConv2D", model_summary=False):
        '''
        Parameter
        -
        - n_depth: the depth of contracting path, i.e. the number of MaxPooling2D layers in total. Defaults to 3
        - kernel_size: Defualts to 5.
        TODO 其实也可以用3*3的空洞卷积
        - conv: Convolutional layer to be used in model. One of "Conv2D" and "SeparableConv2D". Defaults to 'SeparableConv2D'
        - model_summary: Bool. Defaults to False. Whether to print model summary and plot the topograhpy
        '''
        self.Conv2D = layers.Conv2D if conv == "Conv2D" else layers.SeparableConv2D
        self.model = self.define_model(n_depth, kernel_size, RGB)
        if model_summary:
            self.model.summary()
            plot_model(self.model, type(self).__name__+".png", show_shapes=True)
        self.compiled = False

    def define_model(self, n_depth, kernel_size, RGB):
        '''
        Assume the image has shape (height, width, channels)
        '''
        inputs = Input(shape=(None,None, 3 if RGB else 1))
        # inputs = Input(shape=(128,128,1))
        # todo 当图块尺寸为奇数时，concatenate层报错
        # inputs = Input(shape=(567,567,1))

        # 不能用于图块，容易导致黑色背景的图块变得很白
        # inputs = Rescaling()(inputs)

        contracting = []
        for i in range(n_depth+1):
            block = models.Sequential(name='contracting_'+str(i))
            if i != 0:
                block.add(layers.MaxPooling2D())

            # 第一个卷积层含64个filters，最后一个卷积层含256个filters
            block.add(self.Conv2D(
                2**(i+6), kernel_size, use_bias=False, padding='same'))
            block.add(layers.BatchNormalization())
            block.add(layers.Activation("relu"))

            block.add(self.Conv2D(
                2**(i+6), kernel_size, use_bias=False, padding='same'))
            block.add(layers.BatchNormalization())
            block.add(layers.Activation("relu"))
            contracting.append(block)

        expanding = []
        for i in range(n_depth):
            block = models.Sequential(name='expanding_'+str(i))
            if i != 0:
                # 唯一的卷积层含128个filters
                block.add(self.Conv2D(2**(6+n_depth-i),
                          kernel_size, use_bias=False, padding='same'))
                block.add(layers.BatchNormalization())
                block.add(layers.Activation("relu"))

                block.add(self.Conv2D(2**(6+n_depth-i),
                          kernel_size, use_bias=False, padding='same'))
                block.add(layers.BatchNormalization())
                block.add(layers.Activation("relu"))

            block.add(layers.Conv2DTranspose(2**(6+n_depth-i), 2, strides=2))
            expanding.append(block)

        ending = models.Sequential([
            self.Conv2D(2**6, kernel_size, use_bias=False, padding='same'),
            layers.BatchNormalization(),
            layers.Activation("relu"),

            self.Conv2D(2**6, kernel_size, use_bias=False, padding='same'),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(3 if RGB else 1, 1),
            # Normalize to 01 interval
            # Rescaling()

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
        # clip into 01 interval
        outputs=tf.clip_by_value(outputs,0,1)
        # outputs=layers.Add(name="residual")([outputs,inputs])
        return Model(inputs, outputs)

    def train(self, data_generator, epochs, supervised=False, early_stopping_patience=5, restore_best_weights=True, reduce_lr_patience=2, reduce_lr_factor=0.7, **kwargs):
        '''
        Parameter
        -
        - data_generator: N2VDataGenerator instance
        - epochs: Training epochs

        - early_stopping_patience: argument for callback EarlyStopping
        - restore_best_weights: argument for callback EarlyStopping
        - reduce_lr_patience: argument for callback ReduceLROnPlateau
        - reduce_lr_factor: argument for callback ReduceLROnPlateau
        '''
        if not self.compiled:
            self.compile(supervised)
        monitor="val_loss" if data_generator.config.validation_split else "loss"
        callbacks_list = [callbacks.TensorBoard(log_dir="tensorboard"),
                          callbacks.EarlyStopping(
                              monitor, patience=early_stopping_patience, restore_best_weights=restore_best_weights),
                          callbacks.ModelCheckpoint(
                              f"ckpt/{'supervised' if supervised else 'self_supervised'}/best", monitor, save_best_only=True, save_weights_only=True),
                          callbacks.ReduceLROnPlateau(monitor, factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=0.0001)]            
        print(F"{'SUPERVISED' if supervised else 'SELF-SUPERVISED'} TRAINING BEGINS".center(40,'-'))
        return self.model.fit(data_generator.get_training_batch(supervised=supervised),
                              validation_data=data_generator.get_validation_batch(supervised=supervised),
                              callbacks=callbacks_list,
                              steps_per_epoch=data_generator.steps_per_epoch,
                              validation_steps=data_generator.validation_steps,
                              epochs=epochs)

    def compile(self, supervised, learning_rate=0.0005, momentum=0.1):
        # todo 选择更优的optimizer
        # 选择更优的参数 
        if supervised:
            loss="mse"
            metrics=['mae']
        else:
            loss=self.loss
            metrics=None

        optimizer=optimizers.RMSprop(learning_rate, momentum=momentum)
        self.model.compile(optimizer, loss, metrics)
        self.compiled = True

    @tf.function
    def psnr(self, y_true, y_pred):
        '''
        Parameter
        ---
        y_true: (batch_size, height, width, channels), float32, 01 interval, there's no additional channel as mask
        y_pred: (batch_size, height, width, channels), float32, 01 interval
        '''
        return tf.image.psnr(y_true, y_pred, 1)

    @tf.function
    def ssim(self, y_true, y_pred):
        '''See `psnr`'''
        return tf.image.ssim(y_true, y_pred, 1)
    
    # 不需要加@tf.function吗？我好像之前试验过，自动就是的
    def loss(self, y_true, y_pred):
        '''
        Parameter
        ---
        y_true: (batch_size, height, width, channels+1), float32, 01 interval, the additional channel is mask
        y_pred: (batch_size, height, width, channels), float32, 01 interval
        '''
        coords = y_true[..., -1] == 1
        y_true = y_true[..., :-1]
        squared_difference = tf.reduce_mean(
            tf.square(y_true[coords] - y_pred[coords]))
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

    def evaluate(self, data_generator, divide=1, save_dir=None, batch_size = 1):
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
                self._save_images(restored, save_dir, file_name=y)
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
            new_ssim, new_psnr = self.ssim(y, restored_image), self.psnr(y, restored_image)

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

    def _save_images(self, restored, save_dir, file_name):
        # 输入文件有多少个，输出文件就应该有多少个
        if save_dir:
            restored = np.concatenate(restored)
            # 保存为uint8
            # todo 保存100张图片时，也可能会OOM
            # 可以先单张转为uint8，再concatenate尝试解决
            restored = np.around(restored*255).astype("uint8")
            # 如果是灰度图像，去掉channel轴
            try:
                restored = np.squeeze(restored, axis=-1)
            except ValueError:
                pass

            if not os.path.isdir(save_dir): os.makedirs(save_dir)
            # file_name = f"restored_{file_number}.tif"
            file_name=os.path.splitext(os.path.basename(file_name))[0]+"_restored.tif"
            save_path = os.path.join(save_dir, file_name)
            io.imsave(save_path, restored)
