# -*- coding: utf-8 -*-

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import CardTransition
from kivy.uix.widget import Widget
from kivymd.app import MDApp
from kivymd.icon_definitions import md_icons
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label import MDLabel
from kivymd.uix.list import IconLeftWidget, MDList, OneLineIconListItem
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.tab import MDTabsBase


import ginetex_symbols as gs
from android.permissions import Permission, request_permissions


class NeuralNet:
    model_path = ""
    labels_path = ""
    threshold = 0

    _initialized = False
    _interpreter = None
    _input_details = None
    _output_details = None
    _labels = []

    def __init__(self, model_path: str, labels_path: str, threshold: int = 50) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.threshold = threshold

    def _init_model(self) -> None:
        import tflite_runtime.interpreter as tflite

        self._interpreter = tflite.Interpreter(model_path=self.model_path)
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._interpreter.allocate_tensors()

        with open(file=self.labels_path, mode="r") as f:
            self._labels = [line.strip() for line in f.readlines()]

    def _filter_results(
        self, prediction: numpy.ndarray
    ) -> dict[str, dict[str, dict[str, str]]]:
        end_results = {}

        for label_num, score in enumerate(prediction):
            score = round(score / 100 * threshold)
            if score > self.threshold:
                name = self._labels[label_num]
                entry = {"name": name, "score": score}
                category_name = name.split(sep="_", maxsplit=1)[0]

                category_entries = end_results.get(category_name, None)
                if not category_entries or score > category_entries["score"]:
                    end_results[category_name] = entry

        return end_results

    def make_prediction(self, img: numpy.ndarray) -> numpy.ndarray:
        if not self._initialized:
            self._init_model()

        self._interpreter.set_tensor(self._input_details[0]["index"], [img])
        self._interpreter.invoke()

        prediction = self._interpreter.get_tensor(self._output_details[0]["index"][0])

        return self._filter_results(prediction=prediction)


class App(MDApp):
    nn = None

    transition = CardTransition()

    _permissions = [Permission.CAMERA]

    def __init__(self, nn: NeuralNet, *args, **kwargs) -> None:
        self.nn = nn
        super().__init__(*args, **kwargs)

    def _ensure_permissions(self) -> bool:
        granted = None

        def get_answers(permissions: list[str], answers: list[bool]) -> None:
            granted = all(answers)

        request_permissions(permissions=self._permissions, callback=get_answers)

        return granted

    def _set_icons(self) -> None:
        ginetex_icons = {}

        for category in gs.symbols.values():
            for symbol in category:
                ginetex_icons[symbol] = category[symbol]["icon"]

        md_icons.update(ginetex_icons)

    def _create_tabs(self) -> None:
        class Tab(FloatLayout, MDTabsBase):
            def __init__(self, name: StringProperty, glyph: str, title: str) -> None:
                self.name = name
                self.glyph = glyph
                self.title = title
                super().__init__()

        first_tab = None

        for category, meta in gs.category_meta.items():
            tab = Tab(name=category, glyph=meta["icon"], title=meta["translation"])

            if not first_tab:
                first_tab = tab

            self.root.ids.symbols_list_screen.ids.tabs.add_widget(tab)

        self.populate_tab(instance_tab=first_tab)

    def _create_list_item(self, category: str, symbol: str) -> OneLineIconListItem:
        symbol_item = gs.symbols[category][symbol]

        list_item = OneLineIconListItem(
            text="[size=12sp]{info}[/size]".format(info=symbol_item["text"]),
            font_style="Body1",
            _no_ripple_effect=True,
        )

        list_item.add_widget(IconLeftWidget(icon=symbol, font_name="ginetex.ttf",))

        return list_item

    def _show_error(self) -> bool:
        dialog = MDDialog(
            text="Не було знайдено жодного символу.\n\n\nПереконайтесь, що символи підтримуються сканером та що їх добре видно на фотографії.",
            size_hint=[0.6, 0.8],
            radius=[25, 25, 25, 25],
            padding=[15, 15, 15, 15],
        )
        dialog.ids.container.children[3].halign = "center"

        dialog.open()
        Clock.schedule_once(lambda _: dialog.dismiss(), 4)

        return False

    def _show_prediction(
        self, prediction: dict[str, dict[str, dict[str, str]]]
    ) -> bool:
        results = self.root.ids.results_screen.ids.results

        for category, category_labels in gs.symbols.items():
            if category in prediction:
                section_list = MDList()
                section_list.add_widget(
                    MDLabel(
                        text=gs.category_meta[category]["translation"],
                        font_style="H6",
                        halign="center",
                        size_hint_y=None,
                    )
                )
                results_label = prediction[category]["name"]

                if results_label.endswith("yes"):
                    for label in category_labels:
                        if not label.endswith("no") and label.startswith(
                            results_label.rsplit(sep="_", maxsplit=1)[0]
                        ):
                            section_list.add_widget(
                                self._create_list_item(category=category, symbol=label)
                            )

                elif results_label.endswith(("0", "p")):
                    for label in category_labels:
                        if label.startswith(results_label):
                            section_list.add_widget(
                                self._create_list_item(category=category, symbol=label)
                            )

                else:
                    section_list.add_widget(
                        self._create_list_item(category=category, symbol=results_label)
                    )

                results.add_widget(section_list)

        return True

    def build(self) -> ScreenManager:
        self._set_icons()

        with open("main.kv", encoding="utf-8", mode="r") as f:
            return Builder.load_string(f.read())

    def on_start(self) -> None:
        self._create_tabs()

    def populate_tab(
        self,
        instance_tabs: MDTabs = None,
        instance_tab: Tab = None,
        instance_tab_label: MDLabel = None,
        tab_text: StringProperty = None,
    ) -> None:
        if not instance_tab.populated:
            for symbol in gs.symbols[instance_tab.name]:
                instance_tab.ids.symbols.add_widget(
                    self._create_list_item(category=instance_tab.name, symbol=symbol)
                )
        instance_tab.populated = True

    def show_info(self, *args) -> None:
        dialog = MDDialog(
            title="Інформація",
            text=u"[size=18sp]Власником символів догляду за текстильними виробами, які представлено в цьому застосунку, є міжнародна асоціація по маркуванню текстильних виробів [b]GINETEX[/b]. Всі права зберігаються за правовласником.\n\nСканер може показувати схожі за виглядом символи. Наразі розпізнаються наступні:\n[size=28sp][font=ginetex.ttf]twer89qod\nJasnbvUWE[/font][/size]\n\nРекомендується прати речі якомога рідше та за мінімальної температури води.[/size]",
            size_hint=(0.8, 0.8),
            buttons=[
                MDFlatButton(
                    text="ЗРОЗУМІЛО",
                    text_color=self.theme_cls.primary_color,
                    on_release=lambda _: dialog.dismiss(),
                )
            ],
        )
        dialog.open()

    def prepare_scanner(self) -> bool:
        self._ensure_permissions()
        if not self._ensure_permissions():
            Snackbar(
                text="Функція сканування потребує певних дозволів", duration=1.5
            ).show()

            return False

        return True

    def scan_photo(self, scanner: Widget) -> bool:
        import cv2

        file_name = "photo.jpg"
        scanner.export_to_png(filename=file_name)

        img = cv2.imread("../" + file_name)
        resized_img = cv2.resize(img, (224, 224))

        prediction = self.nn.make_prediction(resized_img)

        return self._show_prediction(prediction) if prediction else self._show_error()


if __name__ == "__main__":

    App(
        nn=NeuralNet(
            model_path="app/model.tflite", labels_path="app/labels.txt", threshold=40
        )
    ).run()
