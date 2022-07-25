class CFG:
    MODEL_SAVE_PATH = "./outs/weight.pth"

    wandb = {"project":"CLIPstyler",
            "group":"test",
            "name":"test",
            "notes":"test"}
    lr = 1e-3
    content_path = "./data/face.jpeg"
    save_inference_path = "./outs/inference.png"
    img_size = 512
    text = "Fire"
    source = "a Photo"
    crop_n = 4
    step = 200
    patch_threshold = 1
    lambda_grob = 100
    lambda_content = 1
    lambda_patch = 0