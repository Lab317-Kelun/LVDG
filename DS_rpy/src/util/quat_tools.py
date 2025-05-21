import sys
import numpy as np
from scipy.spatial.transform import Rotation as R


"""
@note all operations below, of which the return is a vector, return 1-D array, 
      unless multiple inputs are given in vectorized operations
"""


def quat_mean(quat_list, tol=1e-5, max_iter=100):
    """
    计算四元数的平均值
    """
    if isinstance(quat_list[0], list) or isinstance(quat_list[0], np.ndarray):
        quat_list = [R.from_quat(q) for q in quat_list]

    q_avg = quat_list[0]
    for _ in range(max_iter):
        errors = [q * q_avg.inv() for q in quat_list]
        error_matrix = np.array([e.as_rotvec() for e in errors])
        mean_error = error_matrix.mean(axis=0)
        q_avg = R.from_rotvec(mean_error) * q_avg
        if np.linalg.norm(mean_error) < tol:
            break
    return q_avg


def _process_x(x):
    """
    x can be either
        - a single R object
        - a list of R objects
    """

    if isinstance(x, list):
        x = list_to_arr(x)
    elif isinstance(x, R):
        x = x.as_quat()[np.newaxis, :]

    return x



def _process_xy(x, y):
    """
    Transform both x and y into (N by M) np.ndarray and normalize to ensure unit quaternions

    x and y can be either
        - 2 single R objects
        - 1 single R object + 1 list of R objects
        - 2 lists of R objects
    
    Except when both x and y are single R objects, always expand and cast the single R object to meet the same shape
    """
    
    M = 4
    if isinstance(x, R) and isinstance(y, list):
        N = len(y)
        x = np.tile(x.as_quat()[np.newaxis, :], (N,1))
        y = list_to_arr(y)

    elif isinstance(y, R) and isinstance(x, list):
        N = len(x)
        y = np.tile(y.as_quat()[np.newaxis, :], (N,1))
        x = list_to_arr(x)

    elif isinstance(x, list) and isinstance(y, list):
        x = list_to_arr(x)
        y = list_to_arr(y)
    
    elif isinstance(x, R) and isinstance(y, R):
        if x.as_quat().ndim == 1:
            x = x.as_quat()[np.newaxis, :]
        else:
            x = x.as_quat()
        if y.as_quat().ndim == 1:
            y = y.as_quat()[np.newaxis, :]
        else:
            y = y.as_quat()

    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if x.ndim == 1 and y.ndim == 1:
            x = x[np.newaxis, :]
            y = y[np.newaxis, :]
        M = x.shape[1]

    else:
        print("Invalid inputs in quaternion operation")
        sys.exit()

    x = x / np.tile(np.linalg.norm(x, axis=1, keepdims=True), (1,M))
    y = y / np.tile(np.linalg.norm(x, axis=1, keepdims=True), (1,M))
    

    return x,y




def unsigned_angle(x, y):
    """
    Vectorized operation

    @param x is always a 1D array
    @param y is either a 1D array or 2D array of N by M

    note: "If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b; i.e. sum(a[i,:] * b) "
    note: "/" divide operator equivalent to np.divide, performing element-wise division
    note:  np.dot, np.linalg.norm(keepdims=False) and the return angle are 1-D array
    """
    x, y = _process_xy(x, y)

    dotProduct = np.sum(x * y, axis=1)

    angle = np.arccos(np.clip(dotProduct, -1, 1))

    return angle





def riem_log(x, y):
    """
    Vectorized operation

    @param x is the point of tangency
    @param y is either a 1D array or 2D array of N by M


    @note special cases to take care of when x=y and angle(x, y) = pi
    @note IF further normalization needed after adding perturbation?

    - Scenario 1:
        When projecting q_train wrt q_att:
            x is a single R object
            y is a list of R objects
    
    - Scenario 2:
        When projecting each w_train wrt each q_train:
            x is a list of R objects
            y is a list of R objects
    
    - Scenario 3:
        When parallel_transport each projected w_train from respective q_train to q_att:
            x is a list of R objects
            y is a single R object

    - Scenario 4:
        When simulating forward, projecting q_curr wrt q_att:
            x is a single R object
            y is a single R object
    """

    np.seterr(invalid='ignore')

    x, y = _process_xy(x, y)

    N, M = x.shape

    angle = unsigned_angle(x, y) 

    y[angle == np.pi] += 0.001 

    x_T_y = np.tile(np.sum(x * y, axis=1,keepdims=True), (1,M))

    x_T_y_x = x_T_y * x

    u_sca =  np.tile(angle[:, np.newaxis], (1, M))
    u_vec =  (y-x_T_y_x) / np.tile(np.linalg.norm(y-x_T_y_x, axis=1, keepdims=True), (1, M))

    u  = u_sca * u_vec

    
    """
    When y=x, the u should be 0 instead of nan.
    Either of the methods below would work
    """
    # u[np.isnan(u)] = 0
    u[angle == 0] = np.zeros([1, M]) 

    return u


def parallel_transport(x, y, v):
    """
    Vectorized operation
    
    parallel transport a vector u from space defined by x to a new space defined by y

    @param: x original tangent point
    @param: y new tangent point
    @param v vector in tangent space (compatible with both 1-D and 2-D NxM)

    """
    v = _process_x(v)
    log_xy = riem_log(x, y)
    log_yx = riem_log(y, x)
    d_xy = unsigned_angle(x, y)


    # a = np.sum(log_xy * v, axis=1) 
    u = v - (log_xy + log_yx) * np.tile(np.sum(log_xy * v, axis=1, keepdims=True) / np.power(d_xy,2)[:, np.newaxis], (1, 4))


    # Find rows containing NaN values
    nan_rows = np.isnan(u).all(axis=1)

    # Replace NaN rows with zero vectors
    u[nan_rows, :] = np.zeros((1, 4))
 
    return u


def riem_exp(x, v):
    """
    Used during 
         i) running savgol filter
        ii) simulation where x is a rotation object, v is a numpy array
    """

    x = _process_x(x)

    if v.shape[0] == 1:

        v_norm = np.linalg.norm(v)

        if v_norm == 0:
            return x

        y = x * np.cos(v_norm) + v / v_norm * np.sin(v_norm)
    
    else:
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)

        y = np.tile(x, (v_norm.shape[0], 1)) * np.tile(np.cos(v_norm), (1,4)) + v / np.tile(v_norm / np.sin(v_norm), (1,4)) 


    # # Find rows containing NaN values
    # nan_rows = np.isnan(y).all(axis=1)

    # # Replace NaN rows with zero vectors
    # y[nan_rows, :] = np.zeros((1, 4))

    return y





def riem_cov(q_mean, q_list):

    q_list_mean = riem_log(q_mean, q_list)
    scatter = q_list_mean.T @ q_list_mean


    cov = scatter/len(q_list)


    return cov




def canonical_quat(q):
    """
    Force all quaternions to have positive scalar part; necessary to ensure proper propagation in DS
    """
    if (q[-1] < 0):
        return -q
    else:
        return q


def list_to_arr(q_list):
    # 如果输入是包含4个浮点数的列表或数组，则将其直接包装成一个四元数列表
    if isinstance(q_list[0], float) and len(q_list) == 4:
        q_list = [q_list]

    # 如果输入是包含多个四元数的嵌套列表或者数组
    if all(isinstance(el, (list, np.ndarray)) and len(el) == 4 for el in q_list):
        q_arr = np.array(q_list)
        return q_arr

    # 检查并扁平化嵌套列表
    if all(isinstance(el, list) for el in q_list):
        q_list = [item for sublist in q_list for item in sublist]

    N = len(q_list)
    M = 4

    # 打印调试信息
    # print("q_list:", q_list)
    # print("Type of q_list elements:", [type(el) for el in q_list])

    q_arr = np.zeros((N, M))

    for i in range(N):
        element = q_list[i]
        # print(f"Element {i}: {element}")  # 打印每个元素的内容
        if isinstance(element, (list, np.ndarray)) and len(element) == 4:
            # 如果元素是包含4个元素的列表或数组，则将其转换为Rotation对象
            r = R.from_quat(element)
            # 获取四元数表示并赋值给q_arr
            q_arr[i, :] = r.as_quat()
        elif isinstance(element, R):
            # 如果元素是Rotation对象，直接获取四元数表示
            q_arr[i, :] = element.as_quat()
        else:
            raise ValueError(f"Element {i} is not a valid quaternion")

    return q_arr


def list_to_euler(q_list):

    N = len(q_list)
    M = 3

    q_arr = np.zeros((N, M))

    for i in range(N):
        q_arr[i, :] = q_list[i].as_euler('xyz')

        # q_arr[i, :] = canonical_quat(q_list[i].as_quat())

    return q_arr