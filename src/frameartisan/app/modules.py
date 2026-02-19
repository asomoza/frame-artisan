from importlib.resources import files

from frameartisan.modules.generation.generation_module import GenerationModule


TXT2IMG_ICON = files("frameartisan.theme.icons").joinpath("txtvid.png")

MODULES = {
    "Generation": (TXT2IMG_ICON, GenerationModule),
}
