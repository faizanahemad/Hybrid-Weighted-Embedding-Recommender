params = dict(n_dims=64, combining_factor=0.1,
                             collaborative_params=dict(
                                 prediction_network_params=dict(lr=0.5, epochs=30, batch_size=256,
                                                                network_width=128, padding_length=50,
                                                                network_depth=4,
                                                                kernel_l2=1e-5,
                                                                bias_regularizer=0.001, dropout=0.05,
                                                                use_resnet=True),
                                 user_item_params=dict(lr=0.1, epochs=20, batch_size=256, l2=0.001, margin=1.0)))