from LED import *
from math import pi

set_orientation(1)
W, H = get_width_adjusted(), get_height_adjusted()

col_n = 5

L = c = h = 0

while True:
    if get_key("l"):
        L += 0.005

    if get_key("."):
        L -= 0.005

    if get_key("k"):
        c += 0.005

    if get_key(","):
        c -= 0.005

    if get_key("j"):
        h += 0.005

    if get_key("m"):
        h -= 0.005

    L = max(0, min(1, L))
    c = max(0, min(1, c))
    h = max(0, min(pi * 2, h))

    print(h, c, L)

    # refresh(color_oklch(L, c, h))

    for i in range(col_n):
        draw_rectangle(
            (W // col_n) * i,
            0,
            W // col_n,
            H,
            color_oklch(L + i / 96, c - i / 96, h + i),
        )

    draw()
