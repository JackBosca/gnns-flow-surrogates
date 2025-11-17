def teacher_forcing_schedule(epoch, epochs, start=1.0, end=0.0):
    """
    Compute the scheduled sampling probability for the given epoch.
    Linear decay from start to end over the epochs.
    Args:
        epoch (int): Current epoch (1-based).
        epochs (int): Total number of epochs.
        start (float): Starting scheduled sampling probability.
        end (float): Ending scheduled sampling probability.
    Returns:
        float: Scheduled sampling probability for the current epoch.
    """
    denom = max(1, epochs - 1)
    
    # linear fraction from 0 (start) to 1 (end)
    frac = (epoch - 1) / denom
    frac = min(1.0, max(0.0, frac))  # clamp to [0, 1]
    
    # linear interpolation between start and end
    tf_prob = max(end, start + (end - start) * frac)
    return tf_prob
