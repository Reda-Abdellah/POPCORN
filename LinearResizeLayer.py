from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np

def infer_spatial_rank(input_tensor):
    """
    e.g. given an input tensor [Batch, X, Y, Z, Feature] the spatial rank is 3
    """
    input_shape = input_tensor.shape
    input_shape.with_rank_at_least(3)
    #dims = input_tensor.get_shape().ndims - 2
    #assert dims > 0, "input tensor should have at least one spatial dim, " \
    #                 "in addition to batch and channel dims"
    return int(input_shape.ndims - 2)

def expand_spatial_params(input_param, spatial_rank, param_type=int):
    """
    Expand input parameter
    e.g., ``kernel_size=3`` is converted to ``kernel_size=[3, 3, 3]``
    for 3D images (when ``spatial_rank == 3``).
    """
    spatial_rank = int(spatial_rank)
    try:
        if param_type == int:
            input_param = int(input_param)
        else:
            input_param = float(input_param)
        return (input_param,) * spatial_rank
    except (ValueError, TypeError):
        pass
    try:
        if param_type == int:
            input_param = \
                np.asarray(input_param).flatten().astype(np.int).tolist()
        else:
            input_param = \
                np.asarray(input_param).flatten().astype(np.float).tolist()
    except (ValueError, TypeError):
        # skip type casting if it's a TF tensor
        pass
    assert len(input_param) >= spatial_rank, \
        'param length should be at least have the length of spatial rank'
    return tuple(input_param[:spatial_rank])	

class LinearResizeLayer(Layer):
	"""
	Resize 2D/3D images using ``tf.image.resize_bilinear``
	(without trainable parameters).
	"""

	def __init__(self, new_size, name='trilinear_resize'):
		"""

		:param new_size: integer or a list of integers set the output
			2D/3D spatial shape.  If the parameter is an integer ``d``,
			it'll be expanded to ``(d, d)`` and ``(d, d, d)`` for 2D and
			3D inputs respectively.
		:param name: layer name string
		"""
		super(LinearResizeLayer, self).__init__(name=name)
		self.new_size = new_size
	
	def compute_output_shape(self, input_shape):
		return (input_shape[0],self.new_size[0],self.new_size[1],self.new_size[2],input_shape[4])	

	def call(self, input_tensor):
		"""
		Resize the image by linearly interpolating the input
		using TF ``resize_bilinear`` function.

		:param input_tensor: 2D/3D image tensor, with shape:
			``batch, X, Y, [Z,] Channels``
		:return: interpolated volume
		"""

		input_spatial_rank = infer_spatial_rank(input_tensor)
		assert input_spatial_rank in (2, 3), \
			"linearly interpolation layer can only be applied to " \
			"2D/3D images (4D or 5D tensor)."
		self.new_size = expand_spatial_params(self.new_size, input_spatial_rank)

		if input_spatial_rank == 2:
			return tf.image.resize_bilinear(input_tensor, self.new_size)

		b_size, x_size, y_size, z_size, c_size = input_tensor.shape.as_list()
		x_size_new, y_size_new, z_size_new = self.new_size

		if (x_size == x_size_new) and (y_size == y_size_new) and (z_size == z_size_new):
			# already in the target shape
			return input_tensor

		# resize y-z
		squeeze_b_x = tf.reshape(input_tensor, [-1, y_size, z_size, c_size])
		resize_b_x = tf.image.resize_bilinear(squeeze_b_x, [y_size_new, z_size_new])
		resume_b_x = tf.reshape(resize_b_x,  [-1, x_size, y_size_new, z_size_new, c_size])

		# resize x
		#   first reorient
		reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
		
		#   squeeze and 2d resize
		squeeze_b_z = tf.reshape(reoriented, [-1, y_size_new, x_size, c_size])
		resize_b_z = tf.image.resize_bilinear(squeeze_b_z, [y_size_new, x_size_new])
		resume_b_z = tf.reshape(resize_b_z, [-1, z_size_new, y_size_new, x_size_new, c_size])

		output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
		return output_tensor
		
		
		
		
