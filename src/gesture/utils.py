import numpy as np


def compute_rotation(landmarks):
    # rotation
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    index = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])

    x_axis = index - wrist
    x_axis /= np.linalg.norm(x_axis)

    y_axis = pinky - wrist
    y_axis /= np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)  # ensure orthogonality

    rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
    return rotation_matrix

def compute_position(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    mean_position = coords.mean(axis=0)
    return mean_position
