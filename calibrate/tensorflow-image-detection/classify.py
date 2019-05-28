import tensorflow as tf
import sys
import os
import tkinter as tk
from tkinter import filedialog
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# image_path = sys.argv[1]

root = tk.Tk()
root.withdraw()

#image_path = sys.argv[1]
image_path = "/root/Documents/nsec19/calibrate/tensorflow-image-detection/image.png"

if image_path:
    
    # Read the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("tf_files/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        while(1):

            os.system('wget --header "Cookie: PHPSESSID=h97fb19ufkl7s4oalf5rdmdtul" -nv -q http://vision.ctf/catchat.php -O /root/Documents/nsec19/calibrate/tensorflow-image-detection/image.png')
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            
            predictions = sess.run(softmax_tensor, \
                    {'DecodeJpeg/contents:0': image_data})
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
            #for node_id in top_k:
                #human_string = label_lines[node_id]
                #score = predictions[0][node_id]
            if predictions[0][0] > predictions[0][1]:
                print("**** Animal: "+str(predictions[0][0]))
                os.system('curl --cookie "PHPSESSID=h97fb19ufkl7s4oalf5rdmdtul" --data "result=animal" http://vision.ctf/index.php -vvvv')
            else:
                print("**** Thing: "+str(predictions[0][1]))
                os.system('curl --cookie "PHPSESSID=h97fb19ufkl7s4oalf5rdmdtul" --data "result=thing" http://vision.ctf/index.php -vvvv')
                #print('%s (score = %.5f)' % (human_string, score))
