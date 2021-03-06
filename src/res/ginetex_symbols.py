# -*- coding: utf-8 -*-

symbols = {
    "WASH": {
        "WASH_no": {"icon": "z", "text": "Прання заборонено",},
        "WASH_hand": {"icon": "t", "text": "Ручне прання, температура до 40°C"},
        "WASH_30": {"icon": "w", "text": "Звичайне прання, температура до 30°C"},
        "WASH_30_mild": {"icon": "e", "text": "Делікатне прання, температура до 30°C"},
        "WASH_30_very_mild": {
            "icon": "r",
            "text": "Особливо делікатне прання, температура до 30°C",
        },
        "WASH_40": {"icon": "8", "text": "Звичайне прання, температура до 40°C"},
        "WASH_40_mild": {"icon": "9", "text": "Делікатне прання, температура до 40°C"},
        "WASH_40_very_mild": {
            "icon": "q",
            "text": "Особливо делікатне прання, температура до 40°C",
        },
        "WASH_60": {"icon": "4", "text": "Звичайне прання, температура до 60°C"},
        "WASH_60_mild": {"icon": "5", "text": "Делікатне прання, температура до 60°C"},
        "WASH_70": {"icon": "3", "text": "Звичайне прання, температура до 70°C"},
        "WASH_95": {"icon": "2", "text": "Звичайне прання, температура до 95°C"},
    },
    "BLEACH": {
        "BLEACH_no": {"icon": "o", "text": "Відбілювання заборонено"},
        "BLEACH_nonchlorine": {
            "icon": "i",
            "text": "Можна відбілювати засобами без вмісту хлору",
        },
        "BLEACH_any": {"icon": "u", "text": "Можна відбілювати будь-якими засобами"},
    },
    "DRY": {
        "DRY_tumble_no": {"icon": "d", "text": "Барабанну сушку заборонено"},
        "DRY_tumble_mild": {
            "icon": "s",
            "text": "Звичайна барабанна сушка, температура до 60°C",
        },
        "DRY_tumble_normal": {
            "icon": "a",
            "text": "Делікатна барабанна сушка, температура до 80°C",
        },
        "DRY_tumble_yes": {"icon": "J", "text": "Барабанну сушку дозволено"},
        "DRY_natural": {"icon": "p", "text": "Звичайну сушку дозволено"},
        "DRY_vertical": {"icon": "f", "text": "Сушка у вертикальному положенні"},
        "DRY_vertical_shade": {
            "icon": "k",
            "text": "Сушка у вертикальному положенні в тіні",
        },
        "DRY_vertical_drip": {
            "icon": "g",
            "text": "Сушка без віджиму у вертикальному положенні",
        },
        "DRY_vertical_drip_shade": {
            "icon": "l",
            "text": "Сушка без віджиму у верт. положенні в тіні",
        },
        "DRY_horizontal": {"icon": "h", "text": "Сушка у горизонтальному положенні"},
        "DRY_horizontal_shade": {
            "icon": "y",
            "text": "Сушка у горизонтальному положенні в тіні",
        },
        "DRY_horizontal_drip": {
            "icon": "j",
            "text": "Сушка без віджиму у горизонтальному положенні",
        },
        "DRY_horizontal_drip_shade": {
            "icon": "x",
            "text": "Сушка без віджиму у гор. положенні в тіні",
        },
    },
    "IRON": {
        "IRON_no": {"icon": "m", "text": "Прасування заборонено"},
        "IRON_low": {"icon": "n", "text": "Прасування при температурі до 110°C"},
        "IRON_moderate": {"icon": "b", "text": "Прасування при температурі до 150°C"},
        "IRON_hot": {"icon": "v", "text": "Прасування при температурі до 200°C"},
    },
    "PROF": {
        "PROF_wet_no": {"icon": "A", "text": "Мокру професійну чистку заборонено"},
        "PROF_wet_w": {"icon": "I", "text": "Звичайна мокра професійна чистка"},
        "PROF_wet_w_mild": {"icon": "O", "text": "Делікатна мокра професійна чистка",},
        "PROF_wet_w_very_mild": {
            "icon": "P",
            "text": "Особливо делікатна мокра професійна чистка",
        },
        "PROF_dry_no": {"icon": "U", "text": "Суху професійну чистку заборонено"},
        "PROF_dry_f": {
            "icon": "T",
            "text": "Звичайна суха професійна чистка (вуглеводні)",
        },
        "PROF_dry_f_mild": {
            "icon": "Z",
            "text": "Делікатна суха професійна чистка (вуглеводні)",
        },
        "PROF_dry_p": {
            "icon": "W",
            "text": "Звичайна суха професійна чистка (перхлоретилен)",
        },
        "PROF_dry_p_mild": {
            "icon": "E",
            "text": "Делікатна суха професійна чистка (перхлоретилен)",
        },
    },
}

category_meta = {
    "WASH": {"icon": "1", "translation": "ПРАННЯ"},
    "BLEACH": {"icon": "u", "translation": "ВІДБІЛЮВАННЯ"},
    "DRY": {"icon": "p", "translation": "СУШКА"},
    "IRON": {"icon": "c", "translation": "ПРАСУВАННЯ"},
    "PROF": {"icon": "Q", "translation": "ХІМЧИСТКА"},
}
