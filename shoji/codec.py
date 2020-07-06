import blosc
import numpy as np
import pickle


class Codec:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype if dtype != "object" else "string"
    
    def encode(self, data: np.ndarray) -> bytes:
        if self.dtype == "string":
            return blosc.compress(pickle.dumps(data))
        else:
            return blosc.pack_array(data)
    
    def decode(self, data: bytes) -> np.ndarray:
        if self.dtype == "string":
            return pickle.loads(blosc.decompress(data))
        else:
            return blosc.unpack_array(data)
        