def load_checkpoint(model, weight):

    if model.sdf_head is not None:
        model.load_state_dict(weight['model_state_dict'])
    else:
        weights = weight['model_state_dict']
        new_state_dict = {k: v for k, v in weights.items() if k in model.state_dict().keys()}
        model.load_state_dict(new_state_dict)
    return model


def resume_training(optimizer, weights, good_metric='higher'):
    """resume the parameters of optimizer, start epoch and current best metric

    Args:
        optimizer (torch.optim): optimizer
        weights (dict): the checkpoints
        good_metric (str, optional): good metric indicator, define whether the higher/lower value as good. Defaults to 'high'.

    Returns:
        _type_: _description_
    """
    if weights.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(weights["optimizer_state_dict"])
    resume_epoch = weights.get("epoch", 0)
    best_metric = weights.get("best_metric", -100000) if good_metric == 'higher' else weights.get("best_metric", 100000)
    return optimizer, resume_epoch, best_metric
    
    
    
    