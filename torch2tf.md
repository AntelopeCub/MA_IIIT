# **torch2tf**
## A list of methods from pytorch or tensorflow/keras, which have equivalent or similar behavior.
pytorch|tensorflow/keras|behavior|difference
:-:|:-:|:-:|:-:
torch.manual_seed(seed)|tf.random.set_seed(seed)|set random seed|None
torch.cuda.is_available()|tf.config.experimental.list_physical_devices('GPU')|check availability of GPU|return both availability and number of GPUs
torch.cuda.device_count()|ditto|ditto|ditto
torch.cuda.set_device(GPU_index)|tf.device(device_name)|set task on specified GPU|use: `with tf.device`
torch.cuda.current_device()|None|get current device|None
torchvision.datasets.CIFAR10()|tf.keras.datasets.cifar10.load_data()|load build-in dataset|None
torch.nn.CrossEntropyLoss()|tf.keras.losses.CategoricalCrossentropy()|compute crossentropy loss between labels and predictions|None
troch.nn.MSELoss()|tf.keras.losses.MeanSquaredError()|compute mean of squares of errors between labels and predictions|None
torch.nn.DataParallel()|tf.distribute.MirroredStrategy()|set data parallelism|use: `with tf.distribute.MirroredStrategy().scope()`
torch.cat()|tf.concat|concatenates tensors along one dimension|None
torch.FloatTensor()|tf.convert_to_tensor()|convert list or numpy array to tensor|None
torch.Tensor.copy_|tf.identity()|copy one tensor|None
torch.norm|tf.norm|compute the norm of vectors, matrices, and tensors|None
torch.randn()|tf.random.normal()|output random values from normal distribution|None
