# Beta functions
# grad=new gradient, grad0=old, dir=search direction
# From A SURVEY OF NONLINEAR CONJUGATE GRADIENT METHODS
def _betaFR(grad, grad0, dir, logger) -> float:
    """Fletcher and Reeves method"""
    # betaFR = sum(dprod(g,g))/sum(dprod(g0,g0))
    dot_grad = grad.dot(grad)
    dot_grad0 = grad0.dot(grad0)
    if dot_grad0 == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of previous gradient is zero!!!")
    else:
        beta = dot_grad / dot_grad0
    return beta


def _betaPRP(grad, grad0, dir, logger) -> float:
    """Polak, Ribiere, Polyak method"""
    # betaPRP = sum(dprod(g,g-g0))/sum(dprod(g0,g0))
    tmp1 = grad.clone()
    # g-g0
    tmp1.scaleAdd(grad0, 1.0, -1.0)
    dot_num = tmp1.dot(grad)
    dot_grad0 = grad0.dot(grad0)
    if dot_grad0 == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of previous gradient is zero!!!")
    else:
        beta = dot_num / dot_grad0
    return beta


def _betaHS(grad, grad0, dir, logger) -> float:
    """Hestenes and Stiefel"""
    # betaHS = sum(dprod(g,g-g0))/sum(dprod(d,g-g0))
    tmp1 = grad.clone()
    # g-g0
    tmp1.scaleAdd(grad0, 1.0, -1.0)
    dot_num = tmp1.dot(grad)
    dot_denom = tmp1.dot(dir)
    if dot_denom == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
    else:
        beta = dot_num / dot_denom
    return beta


def _betaCD(grad, grad0, dir, logger) -> float:
    """Conjugate Descent"""
    # betaCD = -sum(dprod(g,g))/sum(dprod(d,g0))
    dot_num = grad.dot(grad)
    dot_denom = -grad0.dot(dir)
    if dot_denom == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
    else:
        beta = dot_num / dot_denom
    return beta


def _betaLS(grad, grad0, dir, logger) -> float:
    """Liu and Storey"""
    # betaLS = -sum(dprod(g,g-g0))/sum(dprod(d,g0))
    tmp1 = grad.clone()
    # g-g0
    tmp1.scaleAdd(grad0, 1.0, -1.0)
    dot_num = tmp1.dot(grad)
    dot_denom = -grad0.dot(dir)
    if dot_denom == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
    else:
        beta = dot_num / dot_denom
    return beta


def _betaDY(grad, grad0, dir, logger) -> float:
    """Dai and Yuan"""
    # betaDY = sum(dprod(g,g))/sum(dprod(d,g-g0))
    tmp1 = grad.clone()
    # g-g0
    tmp1.scaleAdd(grad0, 1.0, -1.0)
    dot_num = grad.dot(grad)
    dot_denom = tmp1.dot(dir)
    if dot_denom == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
    else:
        beta = dot_num / dot_denom
    return beta


def _betaBAN(grad, grad0, dir, logger) -> float:
    """Bamigbola, Ali and Nwaeze"""
    # betaDY = sum(dprod(g,g-g0))/sum(dprod(g0,g-g0))
    tmp1 = grad.clone()
    # g-g0
    tmp1.scaleAdd(grad0, 1.0, -1.0)
    dot_num = tmp1.dot(grad)
    dot_denom = tmp1.dot(grad0)
    if dot_denom == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
    else:
        beta = -dot_num / dot_denom
    return beta


def _betaHZ(grad, grad0, dir, logger) -> float:
    """Hager and Zhang"""
    # betaN = sum(dprod(g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0)),g))/sum(dprod(d,g-g0))
    tmp1 = grad.clone()
    # g-g0
    tmp1.scaleAdd(grad0, 1.0, -1.0)
    # sum(dprod(g-g0,g-g0))
    dot_diff_g_g0 = tmp1.dot(tmp1)
    # sum(dprod(d,g-g0))
    dot_dir_diff_g_g0 = tmp1.dot(dir)
    if dot_dir_diff_g_g0 == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
    else:
        # g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0))
        tmp1.scaleAdd(dir, 1.0, -2.0 * dot_diff_g_g0 / dot_dir_diff_g_g0)
        # sum(dprod(g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0)),g))
        dot_num = grad.dot(tmp1)
        # dot_num/sum(dprod(d,g-g0))
        beta = dot_num / dot_dir_diff_g_g0
    return beta


def _betaSD(grad, grad0, dir, logger) -> float:
    """Steepest descent"""
    beta = 0.
    return beta


def _get_beta_func(kind: str = "FR") -> callable:
    kind = kind.upper()
    
    if kind == "FR":
        return _betaFR
    elif kind == "PRP":
        return _betaPRP
    elif kind == "HS":
        return _betaHS
    elif kind == "CD":
        return _betaCD
    elif kind == "LS":
        return _betaLS
    elif kind == "DY":
        return _betaDY
    elif kind == "BAN":
        return _betaBAN
    elif kind == "HZ":
        return _betaHZ
    elif kind == "SD":
        return _betaSD
    else:
        raise ValueError("ERROR! Requested Beta function type not existing")