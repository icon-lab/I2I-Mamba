import torch
import numpy as np

def generate_spiral_indices(H, W):
    # Create an empty list to store the indices in spiral order
    indices = []
    
    left, right, top, bottom = 0, W - 1, 0, H - 1

    while left <= right and top <= bottom:
        # Traverse from left to right
        for i in range(left, right + 1):
            indices.append(top * W + i)
        top += 1

        # Traverse downwards
        for i in range(top, bottom + 1):
            indices.append(i * W + right)
        right -= 1

        if top <= bottom:
            # Traverse from right to left
            for i in range(right, left - 1, -1):
                indices.append(bottom * W + i)
            bottom -= 1

        if left <= right:
            # Traverse upwards
            for i in range(bottom, top - 1, -1):
                indices.append(i * W + left)
            left += 1

    return torch.tensor(indices, dtype=torch.long)

# Example usage
H, W = 64, 64
spiral_indices = generate_spiral_indices(H, W)
column_vector = np.arange(64*64)

matrix = np.zeros((64*64, 64*64), dtype=int)

matrix[column_vector, spiral_indices] = 1

np.save('spiral_eye.npy', matrix)
np.save('despiral_eye.npy', np.transpose(matrix))

spiral_indices_r = spiral_indices.flip(0)
matrix = np.zeros((64*64, 64*64), dtype=int)

matrix[column_vector, spiral_indices_r] = 1

np.save('despiral_r_eye.npy', np.transpose(matrix))

