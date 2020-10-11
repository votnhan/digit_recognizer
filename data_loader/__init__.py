from .mnist_dataset import MNISTDataset
from torchvision import transforms as tf

transforms = tf.Compose([
     tf.ToTensor(),
     tf.Normalize((0.1307,), (0.3081,))
])
