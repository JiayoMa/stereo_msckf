import numpy as np
import torch

# Global device configuration for GPU acceleration
# Set to 'cuda' if GPU is available, otherwise falls back to 'cpu'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64  # Use float64 for numerical precision in MSCKF


def to_tensor(arr, device=None, dtype=None):
    """
    Convert numpy array or list to PyTorch tensor.
    
    Arguments:
        arr: numpy array, list, or tensor to convert
        device: target device (default: DEVICE)
        dtype: target dtype (default: DTYPE)
    
    Returns:
        PyTorch tensor on the specified device
    """
    if device is None:
        device = DEVICE
    if dtype is None:
        dtype = DTYPE
    
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    elif isinstance(arr, np.ndarray):
        return torch.from_numpy(arr.astype(np.float64)).to(device=device, dtype=dtype)
    else:
        return torch.tensor(arr, device=device, dtype=dtype)


def to_numpy(tensor):
    """
    Convert PyTorch tensor to numpy array.
    
    Arguments:
        tensor: PyTorch tensor or numpy array
    
    Returns:
        numpy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


# quaternion representation: [x, y, z, w]
# JPL convention


def skew(vec):
    """
    Create a skew-symmetric matrix from a 3-element vector.
    
    Arguments:
        vec: 3-element vector (tensor or numpy array)
    
    Returns:
        3x3 skew-symmetric matrix (tensor)
    """
    vec = to_tensor(vec)
    x, y, z = vec[0], vec[1], vec[2]
    zero = torch.tensor(0., device=vec.device, dtype=vec.dtype)
    return torch.stack([
        torch.stack([zero, -z, y]),
        torch.stack([z, zero, -x]),
        torch.stack([-y, x, zero])
    ])


def to_rotation(q):
    """
    Convert a quaternion to the corresponding rotation matrix.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    
    Arguments:
        q: quaternion [x, y, z, w] (tensor or numpy array)
    
    Returns:
        3x3 rotation matrix (tensor)
    """
    q = to_tensor(q)
    q = q / torch.norm(q)
    vec = q[:3]
    w = q[3]

    identity = torch.eye(3, device=q.device, dtype=q.dtype)
    R = (2*w*w - 1) * identity - 2*w*skew(vec) + 2*torch.outer(vec, vec)
    return R


def to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    
    Arguments:
        R: 3x3 rotation matrix (tensor or numpy array)
    
    Returns:
        quaternion [x, y, z, w] (tensor)
    """
    R = to_tensor(R)
    
    if R[2, 2] < 0:
        if R[0, 0] > R[1, 1]:
            t = 1 + R[0, 0] - R[1, 1] - R[2, 2]
            q = torch.stack([t, R[0, 1] + R[1, 0], R[2, 0] + R[0, 2], R[1, 2] - R[2, 1]])
        else:
            t = 1 - R[0, 0] + R[1, 1] - R[2, 2]
            q = torch.stack([R[0, 1] + R[1, 0], t, R[2, 1] + R[1, 2], R[2, 0] - R[0, 2]])
    else:
        if R[0, 0] < -R[1, 1]:
            t = 1 - R[0, 0] - R[1, 1] + R[2, 2]
            q = torch.stack([R[0, 2] + R[2, 0], R[2, 1] + R[1, 2], t, R[0, 1] - R[1, 0]])
        else:
            t = 1 + R[0, 0] + R[1, 1] + R[2, 2]
            q = torch.stack([R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], t])

    return q / torch.norm(q)


def quaternion_normalize(q):
    """
    Normalize the given quaternion to unit quaternion.
    
    Arguments:
        q: quaternion [x, y, z, w] (tensor or numpy array)
    
    Returns:
        normalized quaternion (tensor)
    """
    q = to_tensor(q)
    return q / torch.norm(q)


def quaternion_conjugate(q):
    """
    Conjugate of a quaternion.
    
    Arguments:
        q: quaternion [x, y, z, w] (tensor or numpy array)
    
    Returns:
        conjugate quaternion (tensor)
    """
    q = to_tensor(q)
    return torch.cat([-q[:3], q[3:]])


def quaternion_multiplication(q1, q2):
    """
    Perform q1 * q2
    
    Arguments:
        q1: first quaternion [x, y, z, w] (tensor or numpy array)
        q2: second quaternion [x, y, z, w] (tensor or numpy array)
    
    Returns:
        product quaternion (tensor)
    """
    q1 = to_tensor(q1)
    q2 = to_tensor(q2)
    q1 = q1 / torch.norm(q1)
    q2 = q2 / torch.norm(q2)

    L = torch.stack([
        torch.stack([q1[3], q1[2], -q1[1], q1[0]]),
        torch.stack([-q1[2], q1[3], q1[0], q1[1]]),
        torch.stack([q1[1], -q1[0], q1[3], q1[2]]),
        torch.stack([-q1[0], -q1[1], -q1[2], q1[3]])
    ])

    q = L @ q2
    return q / torch.norm(q)


def small_angle_quaternion(dtheta):
    """
    Convert the vector part of a quaternion to a full quaternion.
    This function is useful to convert delta quaternion which is  
    usually a 3x1 vector to a full quaternion.
    For more details, check Equation (238) and (239) in "Indirect Kalman 
    Filter for 3D Attitude Estimation: A Tutorial for quaternion Algebra".
    
    Arguments:
        dtheta: 3-element vector (tensor or numpy array)
    
    Returns:
        full quaternion [x, y, z, w] (tensor)
    """
    dtheta = to_tensor(dtheta)
    dq = dtheta / 2.
    dq_square_norm = torch.dot(dq, dq)

    if dq_square_norm <= 1:
        q = torch.cat([dq, torch.sqrt(1 - dq_square_norm).unsqueeze(0)])
    else:
        q = torch.cat([dq, torch.tensor([1.], device=dtheta.device, dtype=dtheta.dtype)])
        q = q / torch.sqrt(1 + dq_square_norm)
    return q


def from_two_vectors(v0, v1):
    """
    Rotation quaternion from v0 to v1.
    
    Arguments:
        v0: source vector (tensor or numpy array)
        v1: target vector (tensor or numpy array)
    
    Returns:
        quaternion representing rotation from v0 to v1 (tensor)
    """
    v0 = to_tensor(v0)
    v1 = to_tensor(v1)
    v0 = v0 / torch.norm(v0)
    v1 = v1 / torch.norm(v1)
    d = torch.dot(v0, v1)

    # if dot == -1, vectors are nearly opposite
    if d < -0.999999:
        axis = torch.linalg.cross(torch.tensor([1., 0., 0.], device=v0.device, dtype=v0.dtype), v0)
        if torch.norm(axis) < 0.000001:
            axis = torch.linalg.cross(torch.tensor([0., 1., 0.], device=v0.device, dtype=v0.dtype), v0)
        q = torch.cat([axis, torch.tensor([0.], device=v0.device, dtype=v0.dtype)])
    elif d > 0.999999:
        q = torch.tensor([0., 0., 0., 1.], device=v0.device, dtype=v0.dtype)
    else:
        s = torch.sqrt((1 + d) * 2)
        axis = torch.linalg.cross(v0, v1)
        vec = axis / s
        w = 0.5 * s
        q = torch.cat([vec, w.unsqueeze(0)])
        
    q = q / torch.norm(q)
    return quaternion_conjugate(q)   # hamilton -> JPL



class Isometry3d(object):
    """
    3d rigid transform using PyTorch tensors for GPU acceleration.
    """
    def __init__(self, R, t):
        """
        Arguments:
            R: 3x3 rotation matrix (tensor or numpy array)
            t: 3-element translation vector (tensor or numpy array)
        """
        self.R = to_tensor(R)
        self.t = to_tensor(t)

    def matrix(self):
        """
        Get the 4x4 homogeneous transformation matrix.
        
        Returns:
            4x4 transformation matrix (tensor)
        """
        m = torch.eye(4, device=self.R.device, dtype=self.R.dtype)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    def inverse(self):
        """
        Compute the inverse transformation.
        
        Returns:
            Isometry3d representing the inverse transform
        """
        R_inv = self.R.T
        return Isometry3d(R_inv, -R_inv @ self.t)

    def __mul__(self, T1):
        """
        Compose two transformations.
        
        Arguments:
            T1: another Isometry3d
        
        Returns:
            Isometry3d representing self * T1
        """
        R = self.R @ T1.R
        t = self.R @ T1.t + self.t
        return Isometry3d(R, t)