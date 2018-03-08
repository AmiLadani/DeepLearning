from zipfile import ZipFile
import numpy as np

class DataLoader(object):
    def __init__(self):
        DIR = '../data/'
        pass
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = './data/' + label_filename + '.zip'
        image_zip = './data/' + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels

    #creating mini_batches
    def create_batches(self, images, labels, batch_size = 100):
        randomArray = np.arange(images.shape[0])
        np.random.shuffle(randomArray)
        images, labels = images[randomArray], labels[randomArray]

        batch_images = [images[i:i+batch_size] for i in xrange(0, len(labels), batch_size)]
        batch_labels = [labels[i:i+batch_size] for i in xrange(0, len(labels), batch_size)]

        return batch_images, batch_labels

