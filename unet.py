from tensorflow.keras import layers, models, Model, Input, callbacks
from tensorflow.keras.utils import plot_model
import tensorflow as tf


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
            plot_model(self.model, 'Unet.png', show_shapes=True)

        self.compiled = False

    def define_model(self, n_depth=2):
        '''
        Assume the gray image has shape (height, width, channel)
        Parameters
        ---
        n_depth:    Defaults to 2, it's suitable for image of size about (128, 128). the depth of contracting path, i.e. the number of MaxPooling2D layers in total. 
                    If n_depth is too big, image of small size will not be properly processed
        '''
        inputs = Input(shape=self.config.input_shape)
        inputs = Rescaling()(inputs)

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

        return Model(inputs, outputs)

    def train(self, data_generator):
        '''
        Parameter
        -
        data_generator: N2VDataGenerator instance
        '''
        if not self.compiled:
            self.compile()
        # x, y=next(data_generator)
        # return self.model.fit(x,y,batch_size=32,epochs=1)

        callbacks_list = [callbacks.TensorBoard(log_dir="tensorboard"),
                          callbacks.EarlyStopping(
                              monitor="val_loss", patience=3),
                          callbacks.ModelCheckpoint(
                              filepath="ckpt/best", monitor="val_loss", save_best_only=True, save_weights_only=True),
                          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=0.00001)]

        return self.model.fit(data_generator.get_training_batch(),
                              validation_data=data_generator.get_validation_batch(),
                              callbacks=callbacks_list,
                              #   todo Remove, just to test
                              steps_per_epoch=10,
                              validation_steps=6,
                              epochs=4)
        #   steps_per_epoch=self.config.steps_per_epoch,
        #   validation_steps=self.config.validation_steps,
        #   epochs=self.config.epochs)

    def compile(self, optimizer="rmsprop"):
        # todo 选择更优的optimizer
        # 自定义metric: PSNR SSIM
        self.model.compile(optimizer,
                           loss=self.loss,
                        #    metrics=[]
                           )
        self.compiled = True

    def loss(self, y_true, y_pred):
        coords = y_true[..., -1] == 1
        y_true = tf.cast(y_true[..., :-1], tf.float32)

        # 归一化
        m = tf.reduce_max(y_true)
        n = tf.reduce_min(y_true)
        y_true = (y_true-n)/(m-n)

        squared_difference = tf.reduce_mean(
            tf.square(y_true[coords] - y_pred[coords]), axis=-1)
        return squared_difference

    def predict(self, noisy_images):
        '''
        Return denoised images of shape (batch_size, height, width, channels)
        The image will be normalized into 01 interval.

        Parameter
        ---
        noisy_images: (batch_size, height, width, channels) dtype=float32
        '''
        pass

    def evaluate(self, noisy_images, clean_images):
        '''
        Parameter
        ---
        noisy_images: (batch_size, height, width, channels)
        clean_images: (batch_size, height, width, channels)

        Return
        ---
        (PSNR, SSIM)
        PSNR: (batch_size,)
        SSIM: (batch_size,)
        '''
        assert noisy_images.shape == clean_images.shape, 'noisy_images and clean_images must have the same shape'
        denoised_images = self.predict(noisy_images)
        psnr = tf.image.psnr(denoised_images, clean_images, 1)
        ssim = tf.image.ssim(denoised_images, clean_images, 1)
        return psnr, ssim
