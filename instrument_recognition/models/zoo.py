
def load_model(model_name, output_units=None, dropout=0.5):
    """ loads an instrument detection model
    options:  openl3mlp6144-finetuned, openl3mlp-512,
             openl3mlp-6144, mlp-512, mlp-6144
    """
    if model_name == 'baseline-512':
        from instrument_recognition.models.openl3mlp import OpenL3MLP
        model = OpenL3MLP(embedding_size=512, dropout=dropout, num_output_units=output_units, 
                          sr=48000, pretrained=False)

    elif model_name == 'baseline-6144':
        from instrument_recognition.models.openl3mlp import OpenL3MLP
        model = OpenL3MLP(embedding_size=6144, dropout=dropout, num_output_units=output_units, 
                          sr=48000, pretrained=False)
        
    elif model_name == 'openl3mlp-512':
        from instrument_recognition.models.openl3mlp import OpenL3MLP
        model = OpenL3MLP(embedding_size=512, 
                          dropout=dropout,
                          num_output_units=output_units)

    elif model_name == 'openl3mlp-6144':
        from instrument_recognition.models.openl3mlp import OpenL3MLP
        model = OpenL3MLP(embedding_size=6144, 
                          dropout=dropout,
                          num_output_units=output_units)

    elif model_name == 'mlp-512':
        from instrument_recognition.models.mlp import MLP512
        model = MLP512(dropout, output_units)  

    elif model_name == 'mlp-6144':
        from instrument_recognition.models.mlp import MLP6144
        model = MLP6144(dropout, output_units)

    else:
        raise ValueError(f"couldnt find model name: {model_name}")

    return model
