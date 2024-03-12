def create_config(name, base_config, **kwargs):
    # Create a copy of the base config to avoid mutating the original
    config = dict(base_config)
    config["config_name"] = name  # Update the config_name
    config["config_value"].update(kwargs)  # Update any additional settings
    return config
