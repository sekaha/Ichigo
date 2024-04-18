from LED import *


set_window_scale(3)
set_width(500)
set_height(250)

# set_orientation(1)
W, H = get_width_adjusted(), get_height_adjusted()


while True:
    center_text()
    align_text_bottom()
    set_font(FNT_NORMAL)
    draw_text(
        W // 2,
        H // 2,
        "abcdefghijklmnopqrstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ1243567890ïáčåð",
        CYAN,
    )
    align_text_top()
    set_font(FNT_SMALL)
    draw_text(
        W // 2,
        H // 2,
        "abcdefghijklmnopqrstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ1243567890",
        CYAN,
    )
    draw()

