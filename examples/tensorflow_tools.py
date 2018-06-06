import numpy as np
import tensorflow as tf


def cast(arr, cast_as, lib=np):
    """Cast an `arr` as a new
    dtype.

    Args:
        arr : ndarray, tensor
            An array or tensor.
        cast_as : str, dtype
            Type coercion to perform.
        lib : module
            One of `np`, `tf`.

    Returns:
        arr : ndarray, tensor

    """
    if isinstance(cast_as, str):
        cast_as = getattr(lib, cast_as)
    if lib == np:
        return arr.astype(cast_as)
    elif lib == tf:
        return tf.cast(arr, dtype=cast_as)
    else:
        raise ValueError("`lib` must must be tensorflow or numpy.")


def bool_tensor_like(tensor, kind):
    """Create a tensor of booleans of `kind`
     with the same shape as `tensor`.
    Args:
        tensor : Tensor
            A tensorflow tensor.
        kind : bool
            If True, return a tensor of Trues
            If False, return a tensor of Falses
    Returns:
        Tensor :
            Tensor of shape `tensor` populated with booleans.
    """
    if kind is True:
        return tf.ones_like(tensor) > 0
    elif kind is False:
        return tf.ones_like(tensor) > 2
    else:
        raise ValueError("`kind` must be a boolean.")


def tf_isclose(a, b, tol=1e-8):
    """Check where `a` is close to `b`
    to some tolerance `tol`.
    Args:
        a : tensor
            Some numeric tensor.
        b : int, float
            Comparison number.
        tol : int, float
            Tolerance about `b`.
    Returns:
        Tensor
    """
    if 'int' in str(a.dtype) and not isinstance(tol, int):
        raise ValueError(
            "`tol` must be an int to operate on an integer tensor `a`."
        )
    elif 'float' in str(a.dtype) and not isinstance(tol, float):
        raise ValueError(
            "`tol` must be a float to operate on a floating-point tensor `a`."
        )
    lower = (b - tol) <= a
    upper = a <= (b + tol)
    return tf.logical_and(lower, upper)


def tf_replace(tensor, d, tol=1e-8):
    """Replace all element in `x` which match
    keys in some dictionary `d` with the corresponding
    value.
    Args:
        tensor : tensor
            Some numeric tensor.
        d : dict
            A dictionary of the form
            `{value_in_tensor: replacement_value}`.
        tol : int, float
            Tolerance about `value_in_tensor`.
    Returns:
        out : Tensor
            A tensor of the shape and dtype of the input `tensor`
            with values replaced as prescribed by `d`.
    Examples:
        >>> import tensorflow as tf
        ...
        >>> arr = np.array([0, 4, 0, 4, 99])
        >>> # Adversarial test case where `d` contains keys
        >>> # that map to values that are themselves keys:
        >>> d = {4: 99, 99: 101, 0: 4}
        >>> expected = np.array([4, 99, 4, 99, 101])  # correct result
        ...
        >>> x = tf.convert_to_tensor(arr, dtype=tf.int32)
        >>> with tf.Session() as sess:
        >>>     x = tf_replace(x, d=d, tol=0)
        >>>     print((x.eval() == expected).all())  # True
    """
    static_falses = bool_tensor_like(tensor, kind=False)

    # Track which elements have been mutated. This state
    # tracking block reassignment in cases where keys in `d` map
    # to values that are themselves keys, e.g., `d = {0: 4, 4: 99}`.
    can_change = bool_tensor_like(tensor, kind=True)
    for k, replacement in d.items():
        close = tf_isclose(tensor, b=k, tol=tol)
        tensor = tf.where(
            condition=tf.logical_and(close, can_change),
            x=tf.ones_like(tensor) * replacement, y=tensor
        )
        can_change = tf.where(close, x=static_falses, y=can_change)
    return tensor
