from os.path import join as pjoin

import numpy as np
import os

import torch
from tqdm import tqdm
import os

import scipy.ndimage.filters as filters

# Define a kinematic tree for the skeletal struture
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

kit_raw_offsets = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ]
)

t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]


kit_tgt_skel_id = '03950'

t2m_tgt_skel_id = '000021'

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

_EPS4 = np.finfo(float).eps * 4.0

_FLOAT_EPS = np.finfo(np.float64).eps

n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain
l_idx1, l_idx2 = 5, 8
    # Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22

def convert_and_return(joints):
    

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(joints[0]))
    return process_file(joints, 0.002, tgt_offsets)[0]

# PyTorch-backed implementations
def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qinv_np(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    return qinv(torch.from_numpy(q).float()).numpy()


def qnormalize(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    return q / torch.norm(q, dim=-1, keepdim=True)


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qeuler(q, order, epsilon=0, deg=True, follow_order=True):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise
    resdict = {"x":x, "y":y, "z":z}

    # print(order)
    reslist = [resdict[order[i]] for i in range(len(order))] if follow_order else [x, y, z]
    # print(reslist)
    if deg:
        return torch.stack(reslist, dim=1).view(original_shape) * 180 / np.pi
    else:
        return torch.stack(reslist, dim=1).view(original_shape)


# Numpy-backed implementations

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous().float()
    r = torch.from_numpy(r).contiguous().float()
    return qmul(q, r).numpy()


def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()


def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda().float()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous().float()
        return qeuler(q, order, epsilon).numpy()


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def euler2quat(e, order, deg=True):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.view(-1, 3)

    ## if euler angles in degrees
    if deg:
        e = e * np.pi / 180.

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = torch.stack((torch.cos(x / 2), torch.sin(x / 2), torch.zeros_like(x), torch.zeros_like(x)), dim=1)
    ry = torch.stack((torch.cos(y / 2), torch.zeros_like(y), torch.sin(y / 2), torch.zeros_like(y)), dim=1)
    rz = torch.stack((torch.cos(z / 2), torch.zeros_like(z), torch.zeros_like(z), torch.sin(z / 2)), dim=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.view(original_shape)


def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack((np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.reshape(original_shape)


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_matrix_np(quaternions):
    q = torch.from_numpy(quaternions).contiguous().float()
    return quaternion_to_matrix(q).numpy()


def quaternion_to_cont6d_np(quaternions):
    rotation_mat = quaternion_to_matrix_np(quaternions)
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d


def quaternion_to_cont6d(quaternions):
    rotation_mat = quaternion_to_matrix(quaternions)
    cont_6d = torch.cat([rotation_mat[..., 0], rotation_mat[..., 1]], dim=-1)
    return cont_6d


def cont6d_to_matrix(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat


def cont6d_to_matrix_np(cont6d):
    q = torch.from_numpy(cont6d).contiguous().float()
    return cont6d_to_matrix(q).numpy()


def qpow(q0, t, dtype=torch.float64):
    ''' q0 : tensor of quaternions
    t: tensor of powers
    '''
    q0 = qnormalize(q0)
    theta0 = torch.acos(q0[..., 0])

    ## if theta0 is close to zero, add epsilon to avoid NaNs
    mask = (theta0 <= 10e-10) * (theta0 >= -10e-10)
    theta0 = (1 - mask) * theta0 + mask * 10e-10
    v0 = q0[..., 1:] / torch.sin(theta0).view(-1, 1)

    if isinstance(t, torch.Tensor):
        q = torch.zeros(t.shape + q0.shape)
        theta = t.view(-1, 1) * theta0.view(1, -1)
    else:  ## if t is a number
        q = torch.zeros(q0.shape)
        theta = t * theta0

    q[..., 0] = torch.cos(theta)
    q[..., 1:] = v0 * torch.sin(theta).unsqueeze(-1)

    return q.to(dtype)


def qslerp(q0, q1, t):
    '''
    q0: starting quaternion
    q1: ending quaternion
    t: array of points along the way

    Returns:
    Tensor of Slerps: t.shape + q0.shape
    '''

    q0 = qnormalize(q0)
    q1 = qnormalize(q1)
    q_ = qpow(qmul(q1, qinv(q0)), t)

    return qmul(q_,
                q0.contiguous().view(torch.Size([1] * len(t.shape)) + q0.shape).expand(t.shape + q0.shape).contiguous())


def qbetween(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v = torch.cross(v0, v1)
    w = torch.sqrt((v0 ** 2).sum(dim=-1, keepdim=True) * (v1 ** 2).sum(dim=-1, keepdim=True)) + (v0 * v1).sum(dim=-1,
                                                                                                              keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))


def qbetween_np(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()


def lerp(p0, p1, t):
    if not isinstance(t, torch.Tensor):
        t = torch.Tensor([t])

    new_shape = t.shape + p0.shape
    new_view_t = t.shape + torch.Size([1] * len(p0.shape))
    new_view_p = torch.Size([1] * len(t.shape)) + p0.shape
    p0 = p0.view(new_view_p).expand(new_shape)
    p1 = p1.view(new_view_p).expand(new_shape)
    t = t.view(new_view_t).expand(new_shape)

    return p0 + t * (p1 - p0)

def matrix_to_quat(R) -> torch.Tensor:
    '''
    https://github.com/duolu/pyrotation/blob/master/pyrotation/pyrotation.py
    Convert a rotation matrix to a unit quaternion.
    This uses the Shepperd’s method for numerical stability.
    '''

    # The rotation matrix must be orthonormal

    w2 = (1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2])
    x2 = (1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])
    y2 = (1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])
    z2 = (1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])

    yz = (R[..., 1, 2] + R[..., 2, 1])
    xz = (R[..., 2, 0] + R[..., 0, 2])
    xy = (R[..., 0, 1] + R[..., 1, 0])

    wx = (R[..., 2, 1] - R[..., 1, 2])
    wy = (R[..., 0, 2] - R[..., 2, 0])
    wz = (R[..., 1, 0] - R[..., 0, 1])

    w = torch.empty_like(x2)
    x = torch.empty_like(x2)
    y = torch.empty_like(x2)
    z = torch.empty_like(x2)

    flagA = (R[..., 2, 2] < 0) * (R[..., 0, 0] > R[..., 1, 1])
    flagB = (R[..., 2, 2] < 0) * (R[..., 0, 0] <= R[..., 1, 1])
    flagC = (R[..., 2, 2] >= 0) * (R[..., 0, 0] < -R[..., 1, 1])
    flagD = (R[..., 2, 2] >= 0) * (R[..., 0, 0] >= -R[..., 1, 1])

    x[flagA] = torch.sqrt(x2[flagA])
    w[flagA] = wx[flagA] / x[flagA]
    y[flagA] = xy[flagA] / x[flagA]
    z[flagA] = xz[flagA] / x[flagA]

    y[flagB] = torch.sqrt(y2[flagB])
    w[flagB] = wy[flagB] / y[flagB]
    x[flagB] = xy[flagB] / y[flagB]
    z[flagB] = yz[flagB] / y[flagB]

    z[flagC] = torch.sqrt(z2[flagC])
    w[flagC] = wz[flagC] / z[flagC]
    x[flagC] = xz[flagC] / z[flagC]
    y[flagC] = yz[flagC] / z[flagC]

    w[flagD] = torch.sqrt(w2[flagD])
    x[flagD] = wx[flagD] / w[flagD]
    y[flagD] = wy[flagD] / w[flagD]
    z[flagD] = wz[flagD] / w[flagD]

    # if R[..., 2, 2] < 0:
    #
    #     if R[..., 0, 0] > R[..., 1, 1]:
    #
    #         x = torch.sqrt(x2)
    #         w = wx / x
    #         y = xy / x
    #         z = xz / x
    #
    #     else:
    #
    #         y = torch.sqrt(y2)
    #         w = wy / y
    #         x = xy / y
    #         z = yz / y
    #
    # else:
    #
    #     if R[..., 0, 0] < -R[..., 1, 1]:
    #
    #         z = torch.sqrt(z2)
    #         w = wz / z
    #         x = xz / z
    #         y = yz / z
    #
    #     else:
    #
    #         w = torch.sqrt(w2)
    #         x = wx / w
    #         y = wy / w
    #         z = wz / w

    res = [w, x, y, z]
    res = [z.unsqueeze(-1) for z in res]

    return torch.cat(res, dim=-1) / 2

def cont6d_to_quat(cont6d):
    return matrix_to_quat(cont6d_to_matrix(cont6d))

class Skeleton(object):
    def __init__(self, offset, kinematic_tree, device):
        self.device = device
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().to(device).float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]

    def njoints(self):
        return len(self._raw_offset)

    def offset(self):
        return self._offset

    def set_offset(self, offsets):
        self._offset = offsets.clone().detach().to(self.device).float()

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._parents

    # joints (batch_size, joints_num, 3)
    def get_offsets_joints_batch(self, joints):
        assert len(joints.shape) == 3
        _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[:, i] = torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None] * _offsets[:, i]

        self._offset = _offsets.detach()
        return _offsets

    # joints (joints_num, 3)
    def get_offsets_joints(self, joints):
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            # print(joints.shape)
            _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]

        self._offset = _offsets.detach()
        return _offsets

    # face_joint_idx should follow the order of right hip, left hip, right shoulder, left shoulder
    # joints (batch_size, joints_num, 3)
    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
        assert len(face_joint_idx) == 4
        '''Get Forward Direction'''
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
        # print(across1.shape, across2.shape)

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
            # forward (batch_size, 3)
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        '''Get Root Rotation'''
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        '''Inverse Kinematics'''
        # quat_params (batch_size, joints_num, 4)
        # print(joints.shape[:-1])
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        # print(quat_params.shape)
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_offset_np[chain[j+1]][np.newaxis,...].repeat(len(joints), axis=0)
                # print(u.shape)
                # (batch, 3)
                v = joints[:, chain[j+1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                # print(u.shape, v.shape)
                rot_u_v = qbetween_np(u, v)

                R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:,chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        joints = torch.zeros(quat_params.shape[:-1] + (3,)).to(self.device)
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(quat_params), -1).detach().to(self.device)
            for i in range(1, len(chain)):
                R = qmul(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot(R, offset_vec) + joints[:, chain[i-1]]
        return joints

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d_np(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(cont6d_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix_np(cont6d_params[:, 0])
            else:
                matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
            for i in range(1, len(chain)):
                matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]][..., np.newaxis]
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints

    def forward_kinematics_cont6d(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            # skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
        joints[..., 0, :] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])
            else:
                matR = torch.eye(3).expand((len(cont6d_params), -1, -1)).detach().to(cont6d_params.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]].unsqueeze(-1)
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints

def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def process_file(positions, feet_thre, tgt_offsets):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

