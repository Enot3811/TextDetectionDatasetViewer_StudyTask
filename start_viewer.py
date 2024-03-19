"""Start application."""


from PySide6.QtWidgets import QApplication

from viewer.viewer_modules.viewer_window import ViewerWindow


def main():
    app = QApplication()
    main_win = ViewerWindow()
    main_win.show()
    app.exec()


if __name__ == '__main__':
    main()
