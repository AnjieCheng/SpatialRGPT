# GroundedSAM / GroundedDino / RAM
rm osdsynth/external/Grounded-Segment-Anything/sam_vit_h_4b8939.pth
rm osdsynth/external/Grounded-Segment-Anything/sam_hq_vit_h.pth
rm osdsynth/external/Grounded-Segment-Anything/groundingdino_swint_ogc.pth
rm osdsynth/external/Grounded-Segment-Anything/recognize-anything/ram_swin_large_14m.pth

# Also clean up pycache
find . -type d -name __pycache__ -exec rm -r {} \+
