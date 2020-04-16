params = dict(n_dims=32, use_content=True,
                  collaborative_params=dict(
                      prediction_network_params=dict(lr=0.001, epochs=25, batch_size=512, margin=0.0,
                                                     gcn_layers=2, ncf_layers=2, conv_depth=2,
                                                     gaussian_noise=0.02, ns_proportion=2, kernel_l2=1e-9, ),
                      user_item_params=dict(lr=0.001, epochs=3, margin=0.0,
                                            gcn_layers=2, ncf_layers=2, conv_depth=2,
                                            gcn_kernel_l2=1e-9, gcn_batch_size=512,
                                            ns_proportion=2, num_walks=10, gaussian_noise=0.0025)))

