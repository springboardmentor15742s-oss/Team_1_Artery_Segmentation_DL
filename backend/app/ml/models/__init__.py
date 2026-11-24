"""Model initialization module.

Provides unified interface to build Standard U-Net and Attention U-Net.
"""
try:
    from .unet import UNet, ConvBlock
    from .attention_unet import AttentionUNet, AttentionGate
except ImportError:
    from models.unet import UNet, ConvBlock
    from models.attention_unet import AttentionUNet, AttentionGate


def build_model(name: str = "attention_unet", in_channels: int = 1, out_channels: int = 1, base: int = 16):
    name = name.lower().replace("-", "_")
    if name in ("unet", "standard_unet"):
        return UNet(in_channels=in_channels, out_channels=out_channels, base=base)
    elif name in ("attention_unet", "attentionunet", "att_unet"):
        return AttentionUNet(in_channels=in_channels, out_channels=out_channels, base=base)
    else:
        raise ValueError(f"Unknown model name: '{name}'. Available: ['attention_unet', 'unet']")


__all__ = ["build_model", "UNet", "AttentionUNet", "ConvBlock", "AttentionGate"]
