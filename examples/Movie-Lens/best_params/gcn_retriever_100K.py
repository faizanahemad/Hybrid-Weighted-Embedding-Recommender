params = dict(n_dims=64, use_content=True,
                  collaborative_params=dict(
                      user_item_params=dict(gcn_lr=0.001, gcn_epochs=3, gcn_layers=2,
                                            gcn_kernel_l2=1e-8, gcn_batch_size=512, conv_depth=2,
                                            margin=1.0,
                                            gaussian_noise=0.005)))
