from boltz.inference import predict_structure

predict_structure(
    input_data = "examples/prot.yaml",
    checkpoint_path = "model_checkpoints/boltz2_conf.ckpt",
    target_dir = "test_fold/targets/",
    msa_dir = "./test_fold/msa",
    output_dir = "./output")
