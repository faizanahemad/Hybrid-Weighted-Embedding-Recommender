params = dict(n_dims=64,  use_content=True,
              collaborative_params=dict(lr=0.001, epochs=5, margin=0.0, label_smoothing_alpha=0.1,
                                        gcn_layers=2, ncf_layers=2, conv_depth=2,
                                        kernel_l2=1e-9, batch_size=1024,
                                        gaussian_noise=0.01,
                                        nsh=1.0, ps_proportion=0.0, ps_threshold=0.01,
                                        ns_proportion=1.0, ns_w2v_proportion=0.5, ns_w2v_exponent=0.75))
