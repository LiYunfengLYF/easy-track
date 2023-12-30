def bbox_clip(x, min_value, max_value):
    new_x = max(min_value, min(x, max_value))
    return new_x
