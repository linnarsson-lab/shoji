from enum import IntEnum


class Compartment(IntEnum):
	Tensors = 1
	TensorValues = 2
	TensorIndex = 3
	TensorRowShapes = 4
	Dimensions = 5
