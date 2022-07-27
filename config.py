class CFG:
    MODEL_SAVE_PATH = "./outs/weight.pth"

    wandb = {"project":"Text2LIVE",
            "group":"test",
            "name":"test",
            "notes":"test"}
    lr = 2.5e-3
    content_path = "./data/ship.jpeg"
    save_inference_path = "./outs/inference.png"
    img_size = 512
    text = "Ship on fire"
    screen = "Fire over the green screen"
    ROI = "Ship"
    step = 200
    optimizer_params = {
        "lr":2.5e-3,
        "momentum":0.9,
        "weight_decay":0.01
    }
    lambda_composition = 1
    lambda_screen = 1
    lambda_structure = 2