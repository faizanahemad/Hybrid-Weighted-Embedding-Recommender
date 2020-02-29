params = dict(n_dims=48, combining_factor=0.1,
                             collaborative_params=dict(
                                 prediction_network_params=dict(lr=0.5, epochs=25, batch_size=64,
                                                                network_width=128, padding_length=50,
                                                                network_depth=4,
                                                                kernel_l2=1e-5,
                                                                bias_regularizer=0.001, dropout=0.05,
                                                                use_resnet=True),
                                 user_item_params=dict(lr=0.1, epochs=25, batch_size=64, l2=0.002, margin=1.0)))