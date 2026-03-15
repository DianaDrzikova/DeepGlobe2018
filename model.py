import segmentation_models_pytorch as smp


def get_model(config):

    model = None
    model_name = config.model.name

    if model_name == "unet":
        model = smp.Unet(
            encoder_name=config.model.encoder_name,
            encoder_weights=config.model.encoder_weights,
            in_channels=config.model.in_channels,
            classes=config.model.classes,
        )
    if model_name == "deeplabv3":
        model = smp.DeepLabV3Plus(
            encoder_name=config.model.encoder_name, 
            encoder_weights=config.model.encoder_weights, 
            classes=config.model.classes, 
            activation=config.model.activation,
        )

    return model