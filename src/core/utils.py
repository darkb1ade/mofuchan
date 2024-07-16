def get_dataset_offset(feature_config):
    max_offset = 0
    for _, config in feature_config.items():
        offset = 0
        if isinstance(config, dict):
            if ("window_size_list" in config) and (
                config["window_size_list"] is not None
            ):
                offset += max(config["window_size_list"])
            if ("smoother_win" in config) and (config["smoother_win"] is not None):
                offset += config["smoother_win"]
            if offset > max_offset:
                max_offset = offset
    return max_offset
