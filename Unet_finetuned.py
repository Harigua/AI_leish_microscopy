import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPool2D,Dropout,Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from PIL import Image, ImageOps


def load_dataset():
    '''
    Loading and organizing the dataset paths for images and their corresponding masks for training, validation, and testing.
    
    Returns:
        lists containing paths to the images and masks for each dataset split (train, validation, and test).
    '''
    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x, test_y = [], []

    # Define paths to your datasets (updated based on your new directory structure)
    images_train_path = '/set/train/images'
    images_valid_path = '/set/validation/images'
    images_test_path = '/set/test/images'
    masks_train_path = '/set/train/Masks'
    masks_valid_path = '/set/validation/Masks'
    masks_test_path = '/set/test/Masks'

    # Get filenames of images and masks without extension
    image_train_files = set([os.path.splitext(filename)[0] for filename in os.listdir(images_train_path)])
    image_valid_files = set([os.path.splitext(filename)[0] for filename in os.listdir(images_valid_path)])
    image_test_files = set([os.path.splitext(filename)[0] for filename in os.listdir(images_test_path)])

    mask_train_files = set([os.path.splitext(filename)[0] for filename in os.listdir(masks_train_path)])
    mask_valid_files = set([os.path.splitext(filename)[0] for filename in os.listdir(masks_valid_path)])
    mask_test_files = set([os.path.splitext(filename)[0] for filename in os.listdir(masks_test_path)])

    # Find matching files
    matching_train_files = list(image_train_files.intersection(mask_train_files))
    matching_valid_files = list(image_valid_files.intersection(mask_valid_files))
    matching_test_files = list(image_test_files.intersection(mask_test_files))

    # Append paths to the lists
    for filename in matching_train_files:
        image_path = os.path.join(images_train_path, filename + '.png')
        mask_path = os.path.join(masks_train_path, filename + '.png')
        train_x.append(image_path)
        train_y.append(mask_path)

    for filename in matching_valid_files:
        image_path = os.path.join(images_valid_path, filename + '.png')
        mask_path = os.path.join(masks_valid_path, filename + '.png')
        valid_x.append(image_path)
        valid_y.append(mask_path)

    for filename in matching_test_files:
        image_path = os.path.join(images_test_path, filename + '.png')
        mask_path = os.path.join(masks_test_path, filename + '.png')
        test_x.append(image_path)
        test_y.append(mask_path)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def resize_and_pad(image, target_size=(512, 512), fill_value=0, mode='constant'):
    """
    Resize and pad an image or mask to a target size while maintaining aspect ratio.
    Args:
        image: Input image or mask (numpy array or PIL Image).
        target_size: Tuple (height, width) specifying the target size.
        fill_value: Value to use for padding (default is 0).
        mode: Padding mode, either 'constant' or 'reflect'.
    Returns:
        Resized and padded image or mask as a numpy array.
    """
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
            image = Image.fromarray(image)
        else:  # Grayscale mask
            image = Image.fromarray(image.astype(np.uint8))
    
    original_width, original_height = image.size
    target_height, target_width = target_size
    
    # Resize while maintaining aspect ratio
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    if isinstance(image, Image.Image):
        resized_image = image.resize((new_width, new_height), Image.BILINEAR if mode == 'constant' else Image.NEAREST)
    else:
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR if mode == 'constant' else cv2.INTER_NEAREST)
    
    # Pad to target size
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    pad_left = delta_w // 2
    pad_right = delta_w - pad_left
    pad_top = delta_h // 2
    pad_bottom = delta_h - pad_top
    
    if isinstance(resized_image, Image.Image):
        padded_image = ImageOps.expand(resized_image, (pad_left, pad_top, pad_right, pad_bottom), fill=fill_value)
        return np.array(padded_image)
    else:
        return cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, 
                                 cv2.BORDER_CONSTANT if mode == 'constant' else cv2.BORDER_REFLECT, 
                                 value=fill_value)
    

def filter_background(img_patches, mask_patches, threshold=FILTER_THRESHOLD):
    """
    Filter out image patches where the mask has less than a certain percentage of foreground pixels.
    Args:
        img_patches: List or array of image patches.
        mask_patches: List or array of corresponding mask patches.
        threshold: Minimum percentage of foreground pixels required to keep the patch.
    Returns:
        Filtered image and mask patches as numpy arrays.
    """

    filtered_img, filtered_mask = [], []
    for img, mask in zip(img_patches, mask_patches):
        if (mask > 0).mean() >= threshold:  # Adjust threshold as needed
            filtered_img.append(img)
            filtered_mask.append(mask)
    return np.array(filtered_img), np.array(filtered_mask)


def extract_grids(image, mask):
    """
    Split image & mask into 24 grids (4 cols × 6 rows), resize/pad to 512x512
    Handles 1844x2709 → (461w x 451.5h) per grid cell
    """
    h, w = image.shape[:2]
    cols = 4
    rows = 6

    # Calculate split points with remainder handling
    col_widths = [w // cols + (1 if i < w % cols else 0) for i in range(cols)]
    row_heights = [h // rows + (1 if i < h % rows else 0) for i in range(rows)]

    img_patches, mask_patches = [], []
    
    current_y = 0
    for row in range(rows):
        current_x = 0
        for col in range(cols):
            # Extract patch
            y_end = current_y + row_heights[row]
            x_end = current_x + col_widths[col]
            
            img_patch = image[current_y:y_end, current_x:x_end]
            mask_patch = mask[current_y:y_end, current_x:x_end]
            
            # Resize and pad to 512x512
            img_patch = resize_and_pad(img_patch, target_size=(512, 512), mode='reflect')
            mask_patch = resize_and_pad(mask_patch, target_size=(512, 512), mode='nearest')
            
            img_patches.append(img_patch)
            mask_patches.append(mask_patch)
            
            current_x += col_widths[col]
        current_y += row_heights[row]
    
    return np.array(img_patches), np.array(mask_patches)


def preprocess(image_path, mask_path):
    """
    Preprocess function to read images, extract patches, filter background, and normalize.
    Args:
        image_path: Path to the image file.
        mask_path: Path to the mask file.
    Returns:
        A TensorFlow dataset containing preprocessed image and mask patches.
    """
    def _py_preprocess(img_path, msk_path):
        """
        Internal function to read images and masks, extract patches, filter background, and normalize.
        Args:
            img_path: Path to the image file.
            msk_path: Path to the mask file.
        Returns:
            Tuple of numpy arrays containing image patches and mask patches.
        """
        # 1. Read original-size images
        image = cv2.imread(img_path.decode(), cv2.IMREAD_COLOR)
        mask = cv2.imread(msk_path.decode(), cv2.IMREAD_GRAYSCALE)
        
        # 2. Extract patches
        image_patches, mask_patches = extract_grids(image, mask)
        
        # 3. Filter background patches
        image_patches, mask_patches = filter_background(image_patches, mask_patches, threshold=FILTER_THRESHOLD)
        
        # 4. Handle empty patches (skip if no patches remain)
        if len(image_patches) == 0:
            return np.zeros((0, PATCH_SIZE, PATCH_SIZE, 3)), np.zeros((0, PATCH_SIZE, PATCH_SIZE, 4))
        
        # 5. Normalize & One-hot encode
        image_patches = image_patches.astype(np.float32) / 255.
        mask_patches = tf.one_hot(mask_patches, 4).numpy().astype(np.float32)
        
        return image_patches, mask_patches

    # Execute in TensorFlow context
    patches_img, patches_mask = tf.numpy_function(
        _py_preprocess,
        [image_path, mask_path],
        (tf.float32, tf.float32))
    
    # Set dynamic shapes using PATCH_SIZE
    patches_img.set_shape([None, 512, 512, 3])
    patches_mask.set_shape([None, 512, 512, 4])
    
    return tf.data.Dataset.from_tensor_slices((patches_img, patches_mask))


def tf_dataset(x, y, batch_size=BATCH_SIZE):
    """
    Create a TensorFlow dataset from image and mask paths.
    Args:
        x: List of image file paths.
        y: List of mask file paths.
        batch_size: Size of the batches to be created.
    Returns:
        A TensorFlow dataset ready for training.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(len(x), reshuffle_each_iteration=True)
    dataset = dataset.flat_map(preprocess)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
    
def conv_block(input, num_filters,regularization=None):
    """    Convolutional block for U-Net architecture.
    Args:
        input: Input tensor from the previous layer.
        num_filters: Number of filters for the convolutional layers in this block.
        regularization: Regularization strength for the convolutional layers. If None, no regularization is applied.
    Returns:
        A tensor representing the output of the convolutional block.
    """
    x = Conv2D(num_filters, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(regularization))(input)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(regularization))(x)
    x = Activation("relu")(x)
    return x

def encoder_block(input, num_filters):
    """    Encoder block for U-Net architecture.
    Args:
        input: Input tensor from the previous layer.
        num_filters: Number of filters for the convolutional layers in this block.
    Returns:
        A tuple containing the output tensor of the block and the max-pooled tensor.
    """
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters,dropout=None):
    """    Decoder block for U-Net architecture.
    Args:
        input: Input tensor from the previous layer.
        skip_features: Features from the corresponding encoder block to concatenate.
        num_filters: Number of filters for the convolutional layers in this block.
        dropout: Dropout rate to apply after the first convolutional layer.
    If None, no dropout is applied.
    If dropout is specified, it should be a float value between 0 and 1.
    Returns:
        A tensor representing the output of the decoder block.
    """
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
    return x

def build_unet(input_shape):
    """
    Build the U-Net model.
    Args:
        input_shape: Tuple representing the shape of the input images (height, width, channels).
    Returns:
        A Keras Model instance representing the U-Net architecture.
    """
    inputs = Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)


    outputs = Conv2D(4, 1, padding="same", activation='softmax')(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

def dice_loss(prediction, ground_truth, epsilon=1e-6):
    """
    Compute the Dice Loss.

    Args:
        prediction: Tensor of shape (batch_size, height, width, num_classes) - model's softmax output
        ground_truth: Tensor of the same shape - one-hot encoded true labels
        epsilon: Smoothing factor to avoid division by zero

    Returns:
        Dice loss as a scalar tensor
    """
    prediction = tf.cast(prediction, tf.float32)
    ground_truth = tf.cast(ground_truth, tf.float32)

    # Flatten the spatial dimensions
    prediction_flat = tf.reshape(prediction, [-1, tf.shape(prediction)[-1]])
    ground_truth_flat = tf.reshape(ground_truth, [-1, tf.shape(ground_truth)[-1]])

    intersection = tf.reduce_sum(prediction_flat * ground_truth_flat, axis=0)
    union = tf.reduce_sum(prediction_flat, axis=0) + tf.reduce_sum(ground_truth_flat, axis=0)

    dice_score = (2.0 * intersection + epsilon) / (union + epsilon)
    dice_loss = 1.0 - tf.reduce_mean(dice_score)

    return dice_loss

if __name__ == "__main__":
    """ U-Net Patch-based Segmentation Training Script
    This script sets up the U-Net model for training on a dataset of images and masks.
    It includes data loading, preprocessing, model building, and training with callbacks.
    It uses TensorFlow and Keras for model definition and training.
    """
    # Define constants
    FILTER_THRESHOLD = 0.1  # Minimum percentage of foreground pixels to keep a patch
    PATCH_SIZE = 512  # Size of the patches to extract from the images

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Hyperparameters """
    lr = 1e-4
    BATCH_SIZE = 8

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset()
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")


    """ Dataset Pipeline """
    train_dataset = tf_dataset(train_x, train_y, batch_size=BATCH_SIZE)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=BATCH_SIZE)

    """ Model """
    model = build_unet(input_shape=(512, 512, 3))
    model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=dice_loss
    )
        
    model_filename = f"Unet.h5"

    # Create callbacks
    callbacks = [EarlyStopping(patience=12, verbose=1),
                ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, verbose=1),
                ModelCheckpoint(model_filename,
                                verbose=1,
                                save_best_only=True)]
    model.fit(train_dataset,
                    epochs=100,
                    callbacks=callbacks,
                    validation_data=valid_dataset)
