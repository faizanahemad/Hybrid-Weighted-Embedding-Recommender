params = dict(n_dims=64, use_content=True,
                  collaborative_params=dict(enable_gcn=True,
                                            lr=0.001, epochs=5, margin=0.0,
                                            layers=3, conv_depth=2,
                                            kernel_l2=1e-9, batch_size=1024,
                                            ns_proportion=2, gaussian_noise=0.005))
