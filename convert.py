import tensorflow as tf

def convert_to_tflite(h5_model_path, output_tflite_path, quantize=False):
    # Load the .h5 model
    model = tf.keras.models.load_model(h5_model_path)

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Convert the model with quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        output_tflite_path = output_tflite_path.replace('.tflite', '_quantized.tflite')
    else:
        tflite_model = converter.convert()

    # Save the model to disk
    with open(output_tflite_path, 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    h5_model_path = 'model.h5'  # Replace 'your_model.h5' with your .h5 model file path
    output_tflite_path = 'converted_model_q.tflite'  # Output path for the TensorFlow Lite model
    convert_to_tflite(h5_model_path, output_tflite_path, quantize=True)
