params = {-1: dict(n_dims=128, n_content_dims=196,
                  collaborative_params=dict(
                      user_item_params=dict(gcn_lr=0.001, gcn_epochs=5, gcn_layers=3,
                                            gcn_kernel_l2=1e-7, gcn_batch_size=1024, conv_depth=1,
                                            margin=1.0,
                                            node2vec_params=dict(num_walks=150, q=0.75)))),
          0: dict(n_dims=128, n_content_dims=196,
                  collaborative_params=dict(
                      user_item_params=dict(gcn_lr=0.001, gcn_epochs=5, gcn_layers=3,
                                            gcn_kernel_l2=1e-8, gcn_batch_size=512, conv_depth=2,
                                            margin=1.0,
                                            gaussian_noise=0.0025,
                                            node2vec_params=dict(num_walks=150, q=0.75)))),
          }