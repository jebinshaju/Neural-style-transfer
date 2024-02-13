import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
from PIL import Image

# Function to preprocess images for VGG19
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return vgg19.preprocess_input(img_array)

# Function to deprocess images
def deprocess_image(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Function to perform neural style transfer

def style_transfer(content_img, style_img, iterations=100, content_weight=1e4, style_weight=1e-1):
    # Rest of the function remains the same
    # You can keep the rest of your style_transfer function as is
    # Load pre-trained VGG19 model
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # Get output layers corresponding to style and content layers
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block4_conv2'
    outputs = [vgg.get_layer(name).output for name in (style_layers + [content_layer])]

    # Build model
    model = Model(inputs=vgg.input, outputs=outputs)

    # Compute gram matrices for style layers
    def gram_matrix(input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    # Preprocess content and style images
    content_img = vgg19.preprocess_input(content_img)
    style_img = vgg19.preprocess_input(style_img)

    style_features = model(style_img)
    style_grams = [gram_matrix(style_feature) for style_feature in style_features[:len(style_layers)]]

    # Extract content features
    content_features = model(content_img)[len(style_layers):]

    # Initialize result image with content image
    generated_img = tf.Variable(content_img)

    # Optimize
    optimizer = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
    best_loss, best_img = float('inf'), None

    for i in range(iterations):
        with tf.GradientTape() as tape:
            model_outputs = model(generated_img)
            style_output_features = model_outputs[:len(style_layers)]
            content_output = model_outputs[len(style_layers):]

            # Content loss
            content_loss = tf.add_n([tf.reduce_mean(tf.square(content_output[i] - content_features[i]))
                                     for i in range(len(content_output))])

            # Style loss
            style_loss = tf.add_n([tf.reduce_mean(tf.square(gram_matrix(style_output) - style_gram))
                                   for style_output, style_gram in zip(style_output_features, style_grams)])

            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss

        gradients = tape.gradient(total_loss, generated_img)
        optimizer.apply_gradients([(gradients, generated_img)])

        # Update best result
        if total_loss < best_loss:
            best_loss = total_loss
            best_img = generated_img.numpy()

        print(f"Iteration: {i+1}/{iterations}, Loss: {total_loss.numpy()}")

    return deprocess_image(best_img)

# Streamlit app
st.title('Neural Style Transfer with VGG19')

content_image = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_image = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_image and style_image:
    content_img = preprocess_image(content_image)
    style_img = preprocess_image(style_image)

    st.write('Content Image:')
    st.image(content_image, use_column_width=True)

    st.write('Style Image:')
    st.image(style_image, use_column_width=True)

    if st.button('Generate Image'):
        combined_img = style_transfer(content_img, style_img)
        st.write('Combined Image:')
        st.image(combined_img, use_column_width=True)

        st.write('Download Combined Image:')
        st.download_button(
            label="Download",
            data=Image.fromarray(combined_img),
            file_name="combined_image.jpg",
            mime="image/jpeg"

        )
