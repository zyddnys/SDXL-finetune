# SDXL-finetune
Adapted from https://github.com/harubaru/waifu-diffusion/blob/main/trainer/diffusers_trainer.py
With the following additions:
1. SDXL support
2. FP32 for VAE
3. Designed for booru tags

# How to use
Have a dataset of images with caption files end with `.txt`, e.g. `danbooru2021/0000/1000.jpg` and `danbooru2021/0000/1000.txt` \
The content of txt file is something like
```
bad aesthetic,gen:panties,gen:oekaki,char:amano_misao_(battle_programmer_shirase),art:haganemaru_kennosuke,meta:lowres,gen:open_mouth,gen:panty_pull,gen:white_panties,gen:school_uniform,gen:1girl,copy:battle_programmer_shirase,gen:underwear,gen:blush,gen:jaggy_line,gen:long_hair,gen:solo
```
comma seperated tags, `<category>:<tag>`, category <i>can be</i> shorted to save space.
things like aesthetic are taken from [waifu diffusion](https://saltacc.notion.site/WD-1-5-Beta-3-Release-Notes-1e35a0ed1bb24c5b93ec79c45c217f63)

# Stats
44GB of VRAM used for batch size of 2 at bucket resolution 896x896
