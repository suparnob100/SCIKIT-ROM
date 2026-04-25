from skfem import LinearForm

@LinearForm
def l_source(v, w):
    """
    Unit source form:
        ∫ 1 * v dx
    """
    return 1.0 * v