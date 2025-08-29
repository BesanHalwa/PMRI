# Original U-net
# train on fit patient data only

import tensorflow as tf
import numpy as np
import os
import datetime
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
IMG_WIDTH = 256
IMG_HEIGHT = 256
INPUT_CHANNELS = 1  # Single-channel grayscale input
OUTPUT_CHANNELS = 6  # One-hot encoded ROI
BATCH_SIZE = 16
BUFFER_SIZE = 1000
EPOCHS = 100
LAMBDA = 100
CHECKPOINT_DIR = './checkpoints'
LOG_DIR = './logs'

# Load .npy datasets
def load_npy_datasets():
    try:
        path = '/home/besanhalwa/Eshan/project1_PMRI/Data/npy_tech_pmri_no_aug_leftRightSplit/'
        train_images = np.load(path+'train_images.npy')  # Shape: (2348, 256, 256)
        train_roi = np.load(path+'train_masks_hot_encoded.npy')  # Shape: (2348, 256, 256, 6)
        val_images = np.load(path+'val_images.npy')  # Shape: (416, 256, 256)
        val_roi = np.load(path+'val_masks_hot_encoded.npy')  # Shape: (416, 256, 256, 6)
        test_images = np.load(path+'test_images.npy')  # Shape: (416, 256, 256)
        test_roi = np.load(path+'test_masks_hot_encoded.npy')  # Shape: (416, 256, 256, 6)
        
        return (train_images, train_roi), (val_images, val_roi), (test_images, test_roi)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise

# Data preprocessing
def normalize(input_image, real_image):
    input_image = tf.cast(input_image, tf.float32)  # Cast to float32
    input_image = (input_image / 127.5) - 1
    real_image = tf.cast(real_image, tf.float32)
    return input_image, real_image

# Create tf.data datasets
def create_datasets(train_data, val_data, test_data):
    train_images, train_roi = train_data
    val_images, val_roi = val_data
    test_images, test_roi = test_data
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_roi))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_roi))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_roi))
    
    train_dataset = (train_dataset
                     .cache()
                     .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
                     .shuffle(BUFFER_SIZE)
                     .batch(BATCH_SIZE)
                     .prefetch(tf.data.AUTOTUNE))
    
    val_dataset = (val_dataset
                   .cache()
                   .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(BATCH_SIZE)
                   .prefetch(tf.data.AUTOTUNE))
    
    test_dataset = (test_dataset
                    .cache()
                    .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE))
    
    return train_dataset, val_dataset, test_dataset

# Model definitions
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                              kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                          kernel_initializer=initializer, activation='softmax')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

# Loss functions
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# Training setup
generator = Generator()
discriminator = Discriminator()
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Checkpointing
checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

# TensorBoard
summary_writer = tf.summary.create_file_writer(
    os.path.join(LOG_DIR, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
)

@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

def fit(train_ds, val_ds, epochs):
    best_val_loss = float('inf')
    patience = 5
    wait = 0
    
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        start = time.time()
        
        # Training
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        for step, (input_image, target) in enumerate(train_ds):
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, target, step)
            total_gen_loss += gen_total_loss
            total_disc_loss += disc_loss
            
            if (step + 1) % 100 == 0:
                logger.info(f"Step {step + 1}/{steps_per_epoch}: Gen Loss = {gen_total_loss.numpy():.4f}, Disc Loss = {disc_loss.numpy():.4f}")
        
        # Validation
        val_gen_loss = 0.0
        val_steps = 0
        for input_image, target in val_ds:
            gen_output = generator(input_image, training=False)
            disc_generated_output = discriminator([input_image, gen_output], training=False)
            gen_total_loss, _, _ = generator_loss(disc_generated_output, gen_output, target)
            val_gen_loss += gen_total_loss
            val_steps += 1
        
        val_gen_loss /= val_steps
        logger.info(f"Epoch {epoch + 1}: Val Gen Loss = {val_gen_loss:.4f}, Time = {time.time() - start:.2f} sec")
        
        # Logging to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', total_gen_loss / steps_per_epoch, step=epoch)
            tf.summary.scalar('disc_loss', total_disc_loss / steps_per_epoch, step=epoch)
            tf.summary.scalar('val_gen_loss', val_gen_loss, step=epoch)
        
        # Checkpointing: Save best model
        if val_gen_loss < best_val_loss:
            best_val_loss = val_gen_loss
            checkpoint.save(file_prefix=checkpoint_prefix + "_best")
            logger.info(f"Saved best checkpoint for epoch {epoch + 1}")
            wait = 0
        else:
            wait += 1
        
        # Checkpointing: Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix + f"_epoch_{epoch + 1}")
            logger.info(f"Saved periodic checkpoint for epoch {epoch + 1}")
        
        # Early stopping
        if wait >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            break

def main():
    try:
        # Create directories
        Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        
        # Load data
        train_data, val_data, test_data = load_npy_datasets()
        train_ds, val_ds, test_ds = create_datasets(train_data, val_data, test_data)
        
        # Train model
        fit(train_ds, val_ds, EPOCHS)
        
        # Save final model
        generator.save('generator_final.h5')
        discriminator.save('discriminator_final.h5')
        logger.info("Training completed and models saved")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
