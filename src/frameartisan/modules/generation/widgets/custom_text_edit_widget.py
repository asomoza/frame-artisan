from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTextEdit


class CustomTextEditWidget(QTextEdit):
    def __init__(self, parent):
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def insertFromMimeData(self, source):
        self.insertPlainText(source.text())

    def insertTextAtCursor(self, text):
        cursor = self.textCursor()
        cursor.insertText(text)

    def insertTriggerAtCursor(self, text: str, add_comas: bool = False):
        cursor = self.textCursor()
        plain = self.toPlainText()
        n = len(plain)

        sel_start = cursor.selectionStart()
        sel_end = cursor.selectionEnd()
        has_sel = cursor.hasSelection()

        pos = sel_start if has_sel else cursor.position()

        def char_at(i: int) -> str:
            if 0 <= i < n:
                return plain[i]
            return ""

        prev = char_at(pos - 1)
        prevprev = char_at(pos - 2)
        nextc = char_at(sel_end if has_sel else pos)

        prefix = ""
        suffix = ""

        if add_comas:
            if pos > 0:
                if prev == " ":
                    if pos > 1 and prevprev != ",":
                        prefix = ", "
                elif prev != ",":
                    prefix = ", "
                else:
                    prefix = " "

            if nextc not in ("", ",") and not nextc.isspace():
                suffix = ", "
        else:
            if pos > 0 and prev not in ("", " ") and not prev.isspace():
                prefix = " "
            if nextc not in ("", " ") and not nextc.isspace():
                suffix = " "

        insert_text = f"{prefix}{text}{suffix}"

        cursor.beginEditBlock()

        if has_sel:
            cursor.setPosition(sel_start)
            cursor.setPosition(sel_end, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.setPosition(sel_start)
            cursor.insertText(insert_text)
            cursor.setPosition(sel_start + len(insert_text))
        else:
            cursor.setPosition(pos)
            cursor.insertText(insert_text)
            cursor.setPosition(pos + len(insert_text))

        cursor.endEditBlock()
        self.setTextCursor(cursor)
