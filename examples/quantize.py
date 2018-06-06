from examples.tensorflow_tools import cast, tf_replace
import pickle
import numpy as np
from os.path import join, dirname
import tensorflow as tf


class Rescaler(object):
    """Rescale an array to a given range.

    References:
        * https://stackoverflow.com/a/5295202/4898004

    Examples:
        >>> import numpy as np
        ...
        >>> np.random.seed(99)
        >>> x = np.random.normal(size=100)
        >>> r = Rescaler()
        >>> x_r = r.rescale(x, a=-1, b=1)
        >>> print(np.isclose(x, r.inv_rescale(x_r)).all())  # True.

    """

    def __init__(self, **kwargs):
        self._a = kwargs.get('a', None)
        self._b = kwargs.get('b', None)
        self._min = kwargs.get('min', None)
        self._max = kwargs.get('max', None)
        self._frozen = False

    def _get_rescaler_state(self):
        attrs = ('_a', '_b', '_min', '_max')
        return {a: getattr(self, a) for a in attrs}

    def freeze(self):
        """Lock Rescaling Parameters."""
        for k, v in self._get_rescaler_state().items():
            if v is None:
                raise ValueError("`{}` cannot be None".format(k))
        self._frozen = True

    def unfreeze(self):
        """Unlock Rescaling Parameters."""
        self._frozen = False

    def rescale(self, x, a=0, b=1):
        """Rescale `x` to be on [`a`, `b`].

        Args:
            x : ndarray
                A numeric numpy array.
            a : int, float
                Lower bound.
            b : int, float
                Upper bound.

        Returns:
            y : ndarray
                `x` rescaled to be on [`a`, `b`].

        """
        if not self._frozen:
            self._a, self._b = a, b
            # ToDo: does not work with TF Tensors:
            self._min, self._max = x.min(), x.max()
        num = (self._b - self._a) * (x - self._min)
        denom = self._max - self._min
        y = (num / denom) + self._a
        return y

    def inv_rescale(self, y):
        """Inverse `rescale()`.

        Args:
            y : ndarray
                The output of `rescale()`.

        Returns:
            x : ndarray
                The inverse of `y`.

        """
        y = y - self._a  # will create a copy.
        y *= (self._max - self._min)
        y /= (self._b - self._a)
        y += self._min
        x = y
        return x


class MuLawTransform(Rescaler):
    """μ-law Algorithm

    Args:
        mu : bits in decimal notation
            Defaults to 255 (8 bits)

    References:
        * https://en.wikipedia.org/wiki/Μ-law_algorithm
    """

    def __init__(self, mu=255):
        super().__init__()
        self._mu = mu

    def mu_tranf(self, x, lib=np):
        """Apply a μ-law transform.

        Args:
            x : ndarray
                A numeric numpy array.
            lib : module
                Package to use. One of `numpy`, `tensorflow`.

        Returns:
            y : ndarray
                Transform.

        Warnings:
            * `arr` will first be rescale s.t. it lies on [-1, 1].

        """
        x = self.rescale(x, a=-1, b=1)
        num = lib.log(1 + self._mu * lib.abs(x))
        denom = lib.log(1 + self._mu)
        y = lib.sign(x) * (num / denom)
        return y

    def i_mu_tranf(self, y, lib=np):
        """Inverse μ-law transform (inverses `mu_tranf()`).

        Args:
            y : ndarray
                The output of `mu_tranf()`.
            lib : module
                Package to use. One of `numpy`, `tensorflow`.

        Returns:
            x : ndarray
                Transform.
        """
        num = ((1 + self._mu) ** lib.abs(y)) - 1
        denom = self._mu
        x = lib.sign(y) * (num / denom)
        x = self.inv_rescale(x)
        return x


class Quantize(MuLawTransform):
    """Quantize ndarrays.

    Args:
        mu : bits in decimal notation
            Defaults to 255 (8 bits).
        n_bins : int
            Number of bins to create.
            If None, use `mu` + 1 (recommended).

    Examples:
        >>> import numpy as np
        ...
        >>> t = np.random.normal(loc=1, scale=1, size=100)
        >>> qtz = Quantize(mu=255)
        >>> t_q = qtz.quant(t)
        >>> t_i = qtz.iquant(t_q)
        >>> mse = np.mean((t - t_i) ** 2)
        >>> print(mse)
        ...
        ... # Test freezing.
        >>> qtz = Quantize(mu=255)
        >>> x = np.random.normal(size=10)
        >>> y = np.random.normal(size=10) * 99999
        ...
        >>> x_i = qtz.quant(x)
        >>> qtz.freeze()
        >>> _ = qtz.quant(y)
        ...
        >>> print(np.mean((qtz.iquant(x_i) - x)**2))  # ~= 0

    """

    def __init__(self, mu=255, n_bins=None):
        super().__init__(mu=mu)
        self._n_bins = n_bins if n_bins else mu + 1
        self.bins = None

    def save_state(self, path, save=True):
        """Save the state of the class as a pickle object.

        Args:
            path : str
                Output path (no file name)
                File name will be `state.
            save : bool, defaults to False
                If False, return the state instead of saving.

        Warnings:
            The file name will invariably be `state.p`.
            Other file names will be over-witten.

        """
        quantize_state = {
            'bins': self.bins,
        }
        rescaler_state = self._get_rescaler_state()
        state = {**quantize_state, **rescaler_state}
        if save:
            out_path = join(dirname(path), "state.p")
            pickle.dump(state, open(out_path, "wb"))
        else:
            return state

    def load_state(self, path):
        """Load a state for the class from a pickle object.

        Args:
            path : str
                Path to the a pickle object saved by save_state.

        """
        state = pickle.load(open(path, "rb"))
        state['_n_bins'] = len(state.get('bins'))
        for name, value in state.items():
            setattr(self, name, value)

    def quant(self, arr):
        """Quantize an `ndarray` by
        (a) apply a mu-law transform,
        (b) binning into `mu` bins and
        (c) rescaling the output s.t. 0 <= `b_transf` <= 1.

        Args:
            arr : ndarray
                a numeric ndarray.

        Returns :
            b_transf : ndarray
                an ndarray with values on [0, 1].

        """
        # ToDo: does not work with TF Tensors.
        transf = self.mu_tranf(arr)
        if not self._frozen:
            self.bins = np.linspace(
                start=transf.min(), stop=transf.max(), num=self._n_bins
            )
        b_transf = np.digitize(transf, bins=self.bins, right=True).astype(np.float32)
        # Rescale s.t. the output is on [0, 1]. Given that `digitize()` returns
        # integers >= 0, we can simply divided by the largest possible number.
        b_transf /= self._n_bins - 1
        return b_transf

    def _i_binning(self, arr, lib):
        """Inverse the binning."""
        if lib == np:
            return np.vectorize(lambda i: self.bins[i])(arr)
        elif lib == tf:
            d = {i: v for i, v in enumerate(self.bins, start=0)}
            # `arr` = ints, and ints have no need for tolerance.
            return tf_replace(arr, d=d, tol=0)
        else:
            raise ValueError("`lib` must be one of: numpy, tensorflow.")

    def iquant(self, b_transf, lib=np):
        """Inverse of `quant()`.

        Args:
            b_transf : ndarray
                the yeild of `b_transf`.
            lib : module
                Package to use. One of: `numpy`, `tensorflow`.

        Returns:
            i_b_transf : ndarray
                the inverse of `b_transf`.
        """
        if not isinstance(self.bins, np.ndarray):
            raise AttributeError("`bin` is not a ndarray.")
        i_b_transf = cast(
            b_transf * (self._n_bins - 1), cast_as='int32', lib=lib
        )
        i_b_transf = self._i_binning(i_b_transf, lib=lib)
        i_b_transf = cast(i_b_transf, cast_as='float32', lib=lib)
        i_b_transf = self.i_mu_tranf(i_b_transf, lib=lib)
        return i_b_transf
