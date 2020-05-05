params = dict(n_dims=64, use_content=True,
              gcn_ncf_params=dict(lr=0.001, epochs=15, batch_size=1024,
                                  gcn_layers=3, ncf_layers=2, conv_depth=1,
                                  ncf_gcn_balance=0.0, label_smoothing_alpha=0.01,
                                  gaussian_noise=0.05, kernel_l2=1e-9,
                                  ps_proportion=0.0,
                                  ns_proportion=2.5, ns_w2v_proportion=0.0, ns_w2v_exponent=0.5))
