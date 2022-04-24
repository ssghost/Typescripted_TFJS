import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers
from IPython import display
import matplotlib.pyplot as plt

class DCGAN:
    def __init__(self):
        self.dsname = ""
        self.images = []
        self.labels = []
        self.dataset = None
        self.dloss = None
        self.gloss = None
        self.generator = None
        self.discriminator = None
        self.goptimizer = tf.keras.optimizers.Adam(1e-4)
        self.doptimizer = tf.keras.optimizers.Adam(1e-4)
        self.seed = tf.random.normal([16, 100])
    
    def load_dataset(self, dsname):
        self.dsname = dsname
        ds = tfds.load(self.dsname, split='train', shuffle_files=True)
        ds = ds.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
        for example in ds.take(1):
            image, label = example["image"], example["label"]
            self.images.append(image)
            self.labels.append(label)
        self.images = self.images.reshape(self.images.shape[0], 28, 28, 1).astype('float32')
        self.images = (self.images - 127.5) / 127.5
        self.dataset = tf.data.Dataset.from_tensor_slices(self.images).shuffle(60000).batch(256)
    
    def make_generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256) 

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        self.generator = model

    def make_discriminator(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        self.discriminator = model

    def cross_entropy(self):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        self.dloss = real_loss + fake_loss

    def generator_loss(self, fake_output):
        self.gloss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def save_check(self):
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.goptimizer,
                                        discriminator_optimizer=self.doptimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
    
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([256, 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.gloss(fake_output)
            disc_loss = self.dloss(real_output, fake_output)

        g_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        d_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.goptimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.doptimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
    

    def train(self, epochs=50):
        for epoch in range(epochs):
            for image_batch in self.dataset:
                self.train_step(image_batch)

            display.clear_output(wait=True)
            self.save_results(self.generator, epoch + 1, self.seed)

            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix = "./ckpt")

        display.clear_output(wait=True)
        self.save_results(self.generator, epochs, self.seed)

    def save_results(self, model, epoch, test_input):
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

    def restore_check(self):
        self.checkpoint.restore(tf.train.latest_checkpoint("./"))

        