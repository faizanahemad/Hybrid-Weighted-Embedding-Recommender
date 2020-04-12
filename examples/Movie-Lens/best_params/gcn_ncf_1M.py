params = dict(n_dims=64, n_content_dims=256,
                  collaborative_params=dict(
                      prediction_network_params=dict(lr=0.001, epochs=20, batch_size=1024, margin=0.0,
                                                     gcn_layers=3, ncf_layers=2, conv_depth=1,
                                                     nsh=1.5, ncf_gcn_balance=1.0, ps_proportion=0.0,
                                                     gaussian_noise=0.01, ns_proportion=1.5, kernel_l2=1e-9, ),
                      user_item_params=dict(lr=0.001, epochs=0, margin=0.0,
                                            gcn_layers=2, ncf_layers=2, conv_depth=2,
                                            gcn_kernel_l2=1e-9, gcn_batch_size=512,
                                            ns_proportion=1, num_walks=0, gaussian_noise=0.0025)))

