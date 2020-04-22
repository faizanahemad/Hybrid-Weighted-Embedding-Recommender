params = dict(n_dims=32, use_content=True,
                  collaborative_params=dict(enable_gcn=True,
                                            lr=0.001, epochs=3, margin=0.0,
                                            layers=2, conv_depth=2,
                                            kernel_l2=1e-9, batch_size=2048,
                                            ns_proportion=2, gaussian_noise=0.0025))