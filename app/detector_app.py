from kivy.app import App

from app.app_grid import MyGrid

class DetectorApp(App):
    def build(self):
        return MyGrid()