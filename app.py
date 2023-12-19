import os
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from IPython.display import Image, display

app = Flask(__name__)
Bootstrap(app)

# Load your model here
loadedModel = tf.keras.models.load_model('classifierModel.h5')

# Define class labels
class_labels = ['Ajrak', 'Balochi Attire', 'Kalash Attire', 'Shalwar Kameez']

# Model-related functions
def preprocess_input(img_array):
    img_array /= 255.0
    return img_array

def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    display(Image(cam_path))

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class='None'
    if request.method == 'POST':
        # Handle the image uploaded by the user
        file = request.files['image']
        if file:
            # Save the image to a temporary file
            img_path = 'temp/temp_image.jpg'
            file.save(img_path)

            # Process the image
            test_image = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            test_image = tf.keras.utils.img_to_array(test_image)
            test_image = test_image.reshape(1, 128, 128, 3)
            predictions = loadedModel.predict(test_image)

            # Display the test image and prediction result
            img = mpimg.imread(img_path)
            plt.imshow(img)
            plt.axis('off')
            predicted_class = class_labels[np.argmax(predictions)]
            # plt.title(predicted_class)
            plt.savefig('static/result_image.png')  # Save the result image

            # Generate and save GradCam
            img_array = preprocess_input(get_img_array(img_path, size=(128, 128)))
            heatmap = make_gradcam_heatmap(img_array, loadedModel, "conv2d_1")
            save_and_display_gradcam(img_path, heatmap, 'static/gradcam.png')

    return render_template('index.html', predicted_class = predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
