
`python generate_chars.py --out_dir ./dataset --font_dir ./fonts --width 30 --height 30 --margin 4 --rotate 30 --rotate_step 1`



gen_char 可能会出现的问题：

1. UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte

请修改 310 行： f=open('src/kz_labels','rb') 。 根据自己的情况 改为 rb 或者 r 

2. NameError: name 'sys' is not defined

import sys

3. 