class mapper:
    def __init__(self, layout):
        self.mapping = {
            k: v
            for k, v in zip(
                layout,
                "`1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:\"ZXCVBNM<>?",
            )
        }

        self.reverse_mapping = {
            k: v
            for k, v in zip(
                "`1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:\"ZXCVBNM<>?",
                layout,
            )
        }

    def map_key(self, c):
        # remap char to the layout
        return self.mapping.get(c, c)

    def decode_str(self, s):
        return "".join([self.reverse_mapping.get(c, c) for c in s])

    def map_str(self, s):
        return "".join([self.map_key(c) for c in s])


# the dataset is frankly not very descriptive of how qwertz should look, so I based it off https://kbdlayout.info/KBDGR?arrangement=ANSI104
mappings = {
    "azerty": mapper(
        "`1234567890-=azertyuiop[]\\qsdfghjkl;'wxcvbnm,./~!@#$%^&*()_+AZERTYUIOP{}|QSDFGHJKL:\"WXCVBNM<>?"
    ),
    "dvorak": mapper(
        "`1234567890[]',.pyfgcrl/=\\aoeuidhtns-;qjkxbmwvz~1234567890{}\"<>PYFGCRL?+|AOEUIDHTNS_:QJKXBMWVZ"
    ),
    "qwerty": mapper(
        "`1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:\"ZXCVBNM<>?"
    ),
    "qwertz": mapper(
        "`1234567890ß´qwertzuiopü+#asdfghjklöäyxcvbnm,.-~!\"§$%&/()=?`QWERTZUIOPÜ*'ASDFGHJKLÖÄYXCVBNM;:_"
    ),
}
