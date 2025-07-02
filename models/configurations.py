vit_teacher_config = {
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "intermediate_size": 3072,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "qkv_bias": True,
    "num_classes": 3
}

swin_teacher_config = {
    "img_size": 224,
    "patch_size": 4,
    "in_chans": 3,
    "num_classes": 3,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.,
    "qkv_bias": True,
    "drop_rate": 0.1
}

student_config = {
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "hidden_size": 384,
    "num_attention_heads": 6,
    "num_hidden_layers": 6,
    "intermediate_size": 1536,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "qkv_bias": True,
    "num_classes": 3
}