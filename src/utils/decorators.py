def provides(*feature_names):
    """
    Tags a function with the DataFrame columns it generates.
    The Engine uses this to dynamically build the feature registry.
    """
    def decorator(func):
        func._provides_features = list(feature_names)
        return func
    return decorator