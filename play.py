
import tensorflow as tf


uniform_init_scale = 0.1
# <tensorflow.python.ops.init_ops.RandomUniform>
uniform_initializer = tf.random_uniform_initializer(
	-uniform_init_scale, 
	uniform_init_scale)

from ops import input_ops
ops import gru_cell

# <tensorflow.python.ops.io_ops.TFRecordReader>
reader = tf.TFRecordReader()

input_file_pattern = './train-?????-of-00001'
shuffle_input_data = True
input_queue_capacity = 50000
num_input_reader_threads = 1

data_files = []
# convert to correct file path for specific OS
data_files.extend(tf.gfile.Glob(input_file_pattern)) # ['.\\train-00000-of-00001']

#  asynchronous programming
# reading data is a lot of waiting
# multiple threads prepare training examples and push them in the queue
# a training thread executes a training op that dequeues mini-batches
# from queue

# a queue to hold the filenames with capacity of 16 filenames at 
# a time
# <tensorflow.python.ops.data_flow_ops.FIFOQueue>
filename_queue = tf.train.string_input_producer(
	data_files, 
	shuffle=shuffle, 
	capacity=16, 
	name="filename_queue"
)
# To construct input pipelines, use the `tf.data` module.

min_after_dequeue = int(0.6 * input_queue_capacity)
# <tensorflow.python.ops.data_flow_ops.RandomShuffleQueue>
values_queue = tf.RandomShuffleQueue(
	capacity=input_queue_capacity,
	min_after_dequeue=min_after_dequeue,
	dtypes=[tf.string],
	shapes=[[]],
	name="random_input_queue"
)

enqueue_ops = []
# for each input reader thread, first read from 
# filename queue, then queue the value (filename) to ShuffleQueue 
for _ in range(num_input_reader_threads):
	_, value = reader.read(filename_queue)
	# Tensor("ReaderReadV2:1", shape=(), dtype=string)
	enqueue_ops.append(values_queue.enqueue([value]))

# Create the queue runner to run the values_queue
tf.train.queue_runner.add_queue_runner(
	tf.train.queue_runner.QueueRunner(
		values_queue, 
		enqueue_ops
	)
)

 
