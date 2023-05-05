import equinox as eqx
from jaxtyping import PyTree
from optax import GradientTransformation

def eqx_init_optimiser(optim: GradientTransformation, params: PyTree) -> PyTree:
    """Initialise an optax optimiser with a given set of parameters.
        Filters out non-array parameters (e.g. functions) using `eqx.is_array`.
    Args:
        optim: The optax optimiser to initialise.
        params: The parameters to initialise the optimiser with.

    Returns:
        The initialised optimiser state.
    """
    return optim.init(eqx.filter(params, eqx.is_array))


