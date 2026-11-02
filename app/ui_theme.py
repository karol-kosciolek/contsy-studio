# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Wyglad aplikacji: jasny, spokojny motyw w stylu aplikacji SaaS i wspolne elementy stron.

Zasady: jasne tlo, biale karty z cienka ramka, czarny glowny przycisk, jeden spokojny akcent
(indygo) tylko dla zaznaczen i paskow postepu, bez gradientow, poswiaty i emoji w menu.
Zawsze jasny motyw - ciemny z neonami wygladal amatorsko; JS zdejmuje klase "dark", ktora
Gradio dodaje przy ciemnym motywie systemu. Czcionka systemowa (Segoe UI Variable na Windows
11) - aplikacja dziala offline.

Uklad: panel boczny (gr.Sidebar) z wyborem modelki i menu w grupach; strony to zakladki gr.Tabs
bez wlasnego paska - menu przelacza je w przegladarce (bez czekania na serwer), wiec zdarzenia
Tab.select dzialaja jak dotad. Pola w kartach (.card) nie maja wlasnych ramek - ramke maja
tylko pola wpisywania; dawniej kazde pole bylo osobnym pudelkiem.
"""
from __future__ import annotations

import html
import inspect
from typing import Iterable, List, Optional, Sequence
from urllib.parse import quote

import gradio as gr

# --- kolory (te same w CSS ponizej) ---
BG, SURFACE, MUTED = "#f7f7f8", "#ffffff", "#f4f4f5"
BORDER, BORDER_2 = "#e4e4e7", "#d4d4d8"
TEXT, TEXT_2, TEXT_3 = "#18181b", "#52525b", "#71717a"
ACCENT, ACCENT_SOFT = "#4f46e5", "#eef2ff"

_COLORS = dict(
    body_background_fill=BG, body_text_color=TEXT, body_text_color_subdued=TEXT_3,
    background_fill_primary=SURFACE, background_fill_secondary=MUTED,
    border_color_primary=BORDER, border_color_accent=BORDER_2,
    color_accent=ACCENT, color_accent_soft=ACCENT_SOFT,
    link_text_color="#4338ca", link_text_color_hover="#3730a3",
    link_text_color_visited="#4338ca", link_text_color_active="#3730a3",
    block_background_fill="transparent", block_border_color="transparent",
    block_label_background_fill="transparent", block_label_text_color="#3f3f46",
    block_title_background_fill="transparent", block_title_text_color="#3f3f46",
    block_info_text_color=TEXT_3, panel_background_fill=SURFACE, panel_border_color=BORDER,
    input_background_fill=SURFACE, input_background_fill_focus=SURFACE,
    input_border_color=BORDER, input_border_color_hover=BORDER_2, input_border_color_focus="#a5b4fc",
    input_placeholder_color="#a1a1aa",
    button_primary_background_fill=TEXT, button_primary_background_fill_hover="#27272a",
    button_primary_text_color="#ffffff", button_primary_text_color_hover="#ffffff",
    button_primary_border_color=TEXT, button_primary_border_color_hover="#27272a",
    button_secondary_background_fill=SURFACE, button_secondary_background_fill_hover=MUTED,
    button_secondary_text_color=TEXT, button_secondary_text_color_hover=TEXT,
    button_secondary_border_color=BORDER, button_secondary_border_color_hover=BORDER_2,
    button_cancel_background_fill=SURFACE, button_cancel_background_fill_hover="#fef2f2",
    button_cancel_text_color="#dc2626", button_cancel_text_color_hover="#b91c1c",
    button_cancel_border_color="#fecaca", button_cancel_border_color_hover="#fca5a5",
    checkbox_background_color=SURFACE, checkbox_background_color_selected=TEXT,
    checkbox_border_color=BORDER_2, checkbox_border_color_selected=TEXT,
    checkbox_border_color_focus="#a5b4fc", checkbox_label_background_fill=SURFACE,
    checkbox_label_background_fill_hover=MUTED, checkbox_label_background_fill_selected=SURFACE,
    checkbox_label_border_color=BORDER, checkbox_label_border_color_selected=BORDER_2,
    checkbox_label_text_color=TEXT_2, checkbox_label_text_color_selected=TEXT,
    slider_color=TEXT, table_border_color=BORDER, table_even_background_fill=SURFACE,
    table_odd_background_fill="#fafafa", table_row_focus=MUTED,
    block_shadow="none", input_shadow="0 1px 2px rgba(24, 24, 27, .04)",
    input_shadow_focus="0 0 0 3px rgba(79, 70, 229, .12)",
    button_primary_shadow="none", button_secondary_shadow="none",
    checkbox_label_shadow="none", shadow_drop="none",
)
_SIZES = dict(
    block_border_width="0px", block_radius="12px", block_padding="0px",
    block_label_border_width="0px", block_label_text_weight="500", block_label_text_size="13px",
    block_title_text_weight="500", block_title_text_size="13px", block_title_padding="0 0 6px 0",
    block_label_padding="0 0 6px 0", block_info_text_size="12.5px",
    input_radius="8px", input_padding="9px 12px", input_text_size="14px",
    input_border_width="1px",
    button_border_width="1px",
    button_large_padding="10px 18px", button_large_radius="8px", button_large_text_size="14px",
    button_large_text_weight="600", button_medium_padding="8px 14px", button_medium_radius="8px",
    button_medium_text_size="14px", button_medium_text_weight="500",
    button_small_padding="6px 12px", button_small_radius="8px", button_small_text_size="13px",
    button_small_text_weight="500", checkbox_border_radius="4px", checkbox_label_padding="6px 12px",
    container_radius="12px", layout_gap="16px", form_gap_width="0px", table_radius="10px",
)


def _theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.indigo, secondary_hue=gr.themes.colors.zinc,
        neutral_hue=gr.themes.colors.zinc, radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_md, text_size=gr.themes.sizes.text_md,
        font=[gr.themes.Font(f) for f in ("Inter", "Segoe UI Variable Text", "Segoe UI",
                                          "system-ui", "-apple-system", "sans-serif")],
        font_mono=[gr.themes.Font(f) for f in ("Cascadia Mono", "Consolas", "monospace")])
    allowed = set(inspect.signature(theme.set).parameters)  # inne wersje Gradio: bez bledu
    theme.set(**{k: v for k, v in {**_COLORS, **_SIZES}.items() if k in allowed})
    # ciemny wariant = jasny (gdyby klasa "dark" zostala na chwile przed skryptem strony)
    for name in list(vars(theme)):
        if name.endswith("_dark") and hasattr(theme, name[:-5]):
            setattr(theme, name, getattr(theme, name[:-5]))
    return theme


THEME = _theme()

# --- ikony menu (24x24, kreska) - jako maska CSS, kolor bierze z tekstu ---
ICONS = {
    "home": '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" '
            'height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/>'
            '<rect x="3" y="16" width="7" height="5" rx="1.5"/>',
    "video": '<rect x="2.5" y="5.5" width="14" height="13" rx="2.5"/>'
             '<path d="m16.5 10.5 5-3v9l-5-3"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2.5"/><circle cx="9" cy="9" r="1.8"/>'
             '<path d="m21 15-4.5-4.5L6 21"/>',
    "plan": '<path d="M11 3.5 12.9 8l4.6 1.9-4.6 1.9L11 16.4l-1.9-4.6L4.5 9.9 9.1 8z"/>'
            '<path d="m18.5 14.5.9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z"/>',
    "persona": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    "packs": '<path d="M21 8 12 3 3 8v8l9 5 9-5z"/><path d="m3 8 9 5 9-5"/><path d="M12 13v8"/>',
    "train": '<circle cx="10" cy="8" r="4"/><path d="M2.5 21a7.5 7.5 0 0 1 12.8-5.3"/>'
             '<path d="M19 14.5v6M16 17.5h6"/>',
    "gallery": '<rect x="7" y="7" width="14" height="14" rx="2.5"/>'
               '<path d="M17 3H5.5A2.5 2.5 0 0 0 3 5.5V17"/><path d="m21 17-3.5-3.5L11 20"/>',
    "queue": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "settings": '<path d="M3 6h11M18 6h3M3 12h4M11 12h10M3 18h13M20 18h1"/>'
                '<circle cx="16" cy="6" r="2"/><circle cx="9" cy="12" r="2"/>'
                '<circle cx="18" cy="18" r="2"/>',
}


def _svg_url(body: str) -> str:
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
           'stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
           f'{body}</svg>')
    return "data:image/svg+xml;utf8," + quote(svg)


BRAND = ("<div class='brand'><div class='logo'><svg viewBox='0 0 24 24' fill='none' "
         "stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
         "<path d='M12 3.5 13.9 9l5.6 1.9-5.6 1.9L12 18.5l-1.9-5.7L4.5 10.9 10.1 9z'/></svg>"
         "</div><div><div class='b-name'>Contsy Studio</div>"
         "<div class='b-sub'>AI media studio</div></div></div>")

CSS = """
:root { --c-bg: #f7f7f8; --c-surface: #fff; --c-muted: #f4f4f5; --c-border: #e4e4e7;
  --c-border-2: #d4d4d8; --c-text: #18181b; --c-text-2: #52525b; --c-text-3: #71717a;
  --c-accent: #4f46e5; --c-accent-soft: #eef2ff; --c-ok: #16a34a; --c-warn: #d97706;
  --c-err: #dc2626; }
body, gradio-app, .gradio-container { background: var(--c-bg) !important; }
.gradio-container { max-width: none !important; width: 100% !important; padding: 0 !important; }
/* lewy odstep przy otwartym panelu ustawia Gradio (--overlap-amount) - tu tylko reszta.
   Bez @media: Gradio dokleja do regul w @media ".contain", wiec nie trafilyby w .wrap */
.gradio-container .main { padding: 0 !important; }
.gradio-container .main > .wrap { padding-top: 28px !important;
  padding-right: clamp(16px, 4vw, 40px) !important; padding-bottom: 48px !important; }
.gradio-container .main > .wrap:not(:has(.sidebar.open)) {
  padding-left: clamp(16px, 4vw, 40px) !important; }
.gradio-container .main > .wrap > .contain { max-width: 1240px; margin: 0 auto; width: 100%; }
footer { display: none !important; }
.prose, .prose p, .prose li { color: var(--c-text-2); }
.prose strong { color: var(--c-text); font-weight: 600; }

/* --- panel boczny --- */
#sidebar { background: var(--c-surface) !important; border-right: 1px solid var(--c-border);
  box-shadow: none !important; }
#sidebar .sidebar-content { padding: 18px 14px 14px !important; height: 100%;
  display: flex; flex-direction: column; }
#sidebar .sidebar-content > .column { flex: 1 1 auto; gap: 14px; min-height: 100%; }
#sidebar .toggle-button { background: var(--c-surface); border-color: var(--c-border); }
.brand { display: flex; align-items: center; gap: 10px; padding: 2px 6px 6px; }
.brand .logo { width: 30px; height: 30px; border-radius: 8px; background: var(--c-text);
  display: flex; align-items: center; justify-content: center; flex: none; }
.brand .logo svg { width: 17px; height: 17px; }
.brand .b-name { font-weight: 650; font-size: 15px; color: var(--c-text); letter-spacing: -0.01em;
  line-height: 1.2; }
.brand .b-sub { font-size: 12px; color: var(--c-text-3); line-height: 1.2; }
#character { padding: 0 4px !important; }
#character .wrap { border-radius: 8px !important; }
#nav { padding: 0 !important; margin-top: 2px; }
#nav .wrap { display: flex; flex-direction: column; gap: 1px; background: none; padding: 0; }
#nav label { position: relative; display: flex; align-items: center; width: 100%;
  padding: 7px 10px !important; border-radius: 8px; border: 0 !important; background: none;
  box-shadow: none !important; color: var(--c-text-2); font-weight: 500; font-size: 14px;
  cursor: pointer; transition: background .12s ease, color .12s ease; }
#nav label:hover { background: #f7f7f8; color: var(--c-text); }
#nav label.selected { background: #efeff1; color: var(--c-text); font-weight: 600; }
#nav label.selected span::before { background-color: var(--c-accent); opacity: 1; }
#nav label input { position: absolute; opacity: 0; width: 1px; height: 1px; pointer-events: none; }
#nav label span { margin: 0 !important; display: flex; align-items: center; gap: 10px; }
#nav label span::before { content: ""; width: 18px; height: 18px; flex: none;
  background-color: currentColor; -webkit-mask: var(--icon) center / contain no-repeat;
  mask: var(--icon) center / contain no-repeat; opacity: .9; }
#nav label::before { position: absolute; top: -21px; left: 10px; font-size: 11px;
  font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #a1a1aa; }
#sb-foot { margin-top: auto !important; border-top: 1px solid var(--c-border); padding-top: 12px;
  gap: 10px !important; flex: 0 0 auto !important; }
.sb-status { font-size: 12.5px; color: var(--c-text-3); line-height: 1.55; padding: 0 4px; }
.sb-status b { color: var(--c-text); font-weight: 600; }
.sb-status .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  margin-right: 6px; vertical-align: 1px; background: var(--c-ok); }
.sb-status .dot.off { background: var(--c-err); } .sb-status .dot.pause { background: var(--c-warn); }
#gpu-pause { width: 100%; }

/* --- strony: zakladki bez wlasnego paska (przelacza je menu) --- */
#pages { border: 0 !important; padding: 0 !important; background: none !important; }
#pages > .tab-wrapper { position: absolute !important; left: -10000px; top: 0; width: 4000px;
  height: 1px; overflow: hidden; }
#pages > .tabitem, #pages .tabitem { border: 0 !important; padding: 0 !important;
  background: none !important; }
.page-head { padding: 0 0 4px; }
.page-head h1 { font-size: 24px; font-weight: 650; letter-spacing: -0.025em; margin: 0;
  color: var(--c-text); line-height: 1.25; }
.page-head p { margin: 4px 0 0; color: var(--c-text-3); font-size: 14px; }
.page-head p b { color: var(--c-text-2); font-weight: 600; }
.section-title { font-size: 15px; font-weight: 600; color: var(--c-text); margin: 0; }
.section-sub { font-size: 13px; color: var(--c-text-3); margin: 2px 0 0; }

/* --- karty: biale, cienka ramka; pola w srodku bez wlasnych ramek --- */
.card { background: var(--c-surface) !important; border: 1px solid var(--c-border) !important;
  border-radius: 12px !important; padding: 20px !important; gap: 16px !important;
  box-shadow: 0 1px 2px rgba(24, 24, 27, .03); }
.card.tight { padding: 14px 16px !important; }
.card > .form, .card .form { border: 0 !important; background: none !important; gap: 16px !important;
  box-shadow: none !important; }
.form { gap: 16px !important; border: 0 !important; background: none !important; }
.block.padded { padding: 0 !important; }
.hint p, .hint li, .hint { font-size: 13px !important; color: var(--c-text-3) !important; }
.status-box p { margin: 2px 0; font-size: 13.5px; color: var(--c-text-2); }
.status-box ul { margin: 6px 0 0; }
.st { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px;
  vertical-align: 1px; }
.st-ok { background: var(--c-ok); } .st-err { background: var(--c-err); }
.st-warn { background: var(--c-warn); } .st-info { background: var(--c-accent); }

/* --- przyciski --- */
button.lg, button.md, button.sm { font-weight: 500; box-shadow: none !important; }
button.primary { border: 1px solid var(--c-text) !important; }
.cta { min-height: 44px !important; font-weight: 600 !important; }
.compact-btn { align-self: flex-start !important; width: auto !important; flex: 0 0 auto !important; }
.toolbar { align-items: center !important; gap: 8px !important; flex-wrap: wrap !important; }
.toolbar > * { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
.toolbar .grow, .toolbar > .form:has(.grow) { flex: 1 1 auto !important; }

/* --- wybory (radio) jako przelacznik segmentowy --- */
fieldset:not(#nav) > span[data-testid="block-info"] { display: block; }
fieldset:not(#nav) .wrap:has(> label > input[type="radio"]) { display: inline-flex;
  flex-wrap: wrap; gap: 2px; padding: 3px; background: var(--c-muted); border-radius: 10px;
  width: auto; }
fieldset:not(#nav) .wrap > label:has(> input[type="radio"]) { background: none !important;
  border: 0 !important; box-shadow: none !important; padding: 6px 12px !important;
  border-radius: 7px !important; color: var(--c-text-2) !important; font-weight: 500;
  font-size: 13.5px; cursor: pointer; }
fieldset:not(#nav) .wrap > label:has(> input[type="radio"]):hover { color: var(--c-text) !important; }
fieldset:not(#nav) .wrap > label:has(> input[type="radio"]).selected { background: #fff !important;
  color: var(--c-text) !important;
  box-shadow: 0 1px 2px rgba(24, 24, 27, .10), 0 0 0 1px rgba(24, 24, 27, .05) !important; }
fieldset:not(#nav) .wrap > label > input[type="radio"] { position: absolute; opacity: 0;
  width: 1px; height: 1px; pointer-events: none; }
fieldset:not(#nav) .wrap > label > input[type="radio"] + span { margin: 0 !important; }

/* --- przelaczniki (checkbox z klasa .switch) --- */
.switch input[type="checkbox"] { appearance: none; -webkit-appearance: none; width: 34px !important;
  height: 20px !important; border-radius: 999px !important; border: 0 !important;
  background-color: var(--c-border-2) !important; background-image: none !important;
  position: relative; cursor: pointer; flex: none; transition: background-color .15s ease; }
.switch input[type="checkbox"]::after { content: ""; position: absolute; top: 2px; left: 2px;
  width: 16px; height: 16px; border-radius: 50%; background: #fff;
  box-shadow: 0 1px 2px rgba(0, 0, 0, .2); transition: transform .15s ease; }
.switch input[type="checkbox"]:checked { background-color: var(--c-text) !important; }
.switch input[type="checkbox"]:checked::after { transform: translateX(14px); }
.switch label { gap: 10px; font-weight: 500; color: var(--c-text); }

/* --- akordeony --- */
.gr-accordion { overflow: visible !important; }
.card .gr-accordion { border-top: 1px solid var(--c-border) !important; border-radius: 0 !important;
  padding-top: 12px !important; }
.card .label-wrap, .accordion-card .label-wrap { font-weight: 600; color: var(--c-text); }
.block.gr-accordion.accordion-card { background: var(--c-surface) !important;
  border: 1px solid var(--c-border) !important; border-radius: 12px !important;
  padding: 14px 20px !important; }

/* --- wskazniki (pulpit, kolejka) --- */
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.kpi { background: var(--c-surface); border: 1px solid var(--c-border); border-radius: 12px;
  padding: 14px 16px; min-width: 0; }
.kpi .k-label { font-size: 12.5px; color: var(--c-text-3); font-weight: 500; }
.kpi .k-value { font-size: 22px; font-weight: 650; letter-spacing: -0.02em; margin-top: 4px;
  color: var(--c-text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi .k-value.small { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; line-height: 1.35;
  white-space: normal; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.kpi .k-sub { font-size: 12.5px; color: var(--c-text-3); margin-top: 3px; }
.bar { height: 6px; border-radius: 999px; background: var(--c-muted); overflow: hidden; margin-top: 10px; }
.bar > div { height: 100%; background: var(--c-accent); border-radius: 999px; transition: width .6s ease; }
.job { background: var(--c-surface); border: 1px solid var(--c-border); border-radius: 12px;
  padding: 16px 18px; }
.job.idle { color: var(--c-text-3); }
.job .j-title { font-weight: 600; color: var(--c-text); }
.job .j-meta { font-size: 13px; color: var(--c-text-3); margin-top: 8px; }

/* --- tabele i plakietki --- */
.tbl-wrap { overflow-x: auto; }
.tbl { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13.5px;
  border: 0 !important; margin: 0 !important; }
.tbl th, .tbl td { border: 0 !important; background: none !important; }
.tbl th { text-align: left; font-weight: 500; color: var(--c-text-3); font-size: 12px;
  padding: 8px 12px; border-bottom: 1px solid var(--c-border) !important; white-space: nowrap; }
.tbl td { padding: 10px 12px; border-bottom: 1px solid #f0f0f2 !important; color: var(--c-text);
  vertical-align: middle; }
.tbl tr:last-child td { border-bottom: 0 !important; }
.tbl .muted { color: var(--c-text-3); }
.tbl .num { font-variant-numeric: tabular-nums; }
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 2px 9px; border-radius: 999px;
  font-size: 12px; font-weight: 500; background: var(--c-muted); color: var(--c-text-2);
  white-space: nowrap; }
.badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.badge.ok { background: #f0fdf4; color: #15803d; } .badge.warn { background: #fffbeb; color: #b45309; }
.badge.err { background: #fef2f2; color: #b91c1c; } .badge.info { background: var(--c-accent-soft);
  color: #4338ca; } .badge.plain::before { display: none; }
.empty { padding: 28px 16px; text-align: center; color: var(--c-text-3); font-size: 13.5px;
  border: 1px dashed var(--c-border-2); border-radius: 12px; background: #fcfcfc; }
.empty b { color: var(--c-text); display: block; font-size: 14.5px; margin-bottom: 4px; }

/* --- galerie --- */
.gallery-container .grid-wrap { padding: 2px !important; overflow-y: auto !important; }
.grid-container { gap: 10px !important; }
.thumbnail-item.thumbnail-lg { border-radius: 10px !important; border: 1px solid var(--c-border) !important;
  box-shadow: none !important; background: var(--c-muted) !important; transition: transform .15s ease,
  box-shadow .15s ease; }
.thumbnail-item.thumbnail-lg:hover { --ring-color: transparent; border-color: var(--c-border-2) !important;
  filter: none !important; transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(24, 24, 27, .10) !important; }
.thumbnail-item.thumbnail-lg:has(video)::before { content: ""; position: absolute; left: 8px; top: 8px;
  z-index: 3; width: 26px; height: 26px; border-radius: 999px; background: rgba(24, 24, 27, .72)
  url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M9 7v10l8-5z' fill='white'/%3E%3C/svg%3E")
  center / 16px no-repeat; }
.caption-label { left: 8px !important; right: auto !important; bottom: 8px !important;
  border: 0 !important; border-radius: 6px !important; background: rgba(255, 255, 255, .92) !important;
  color: var(--c-text-2) !important; font-size: 11.5px !important; padding: 2px 7px !important;
  opacity: 0; transition: opacity .15s ease; }
.thumbnail-lg:hover .caption-label { opacity: 1 !important; }
#studio-tiles .caption-label, #char-cards .caption-label { opacity: 1 !important; font-weight: 600;
  color: var(--c-text) !important; }
#studio-tiles .grid-container, #char-cards .grid-container {
  grid-template-columns: repeat(auto-fill, minmax(150px, 180px)) !important; }
#studio-tiles .grid-wrap { height: auto !important; min-height: 0 !important; max-height: 440px; }
#studio-recent .grid-container, #hist-gallery .grid-container {
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr)) !important; }
.gallery-container .icon-button-wrapper { background: none !important; box-shadow: none !important; }
video { border-radius: 10px; }
.result-card .block label[data-testid="block-label"] { display: none; }

/* Historia: zaznaczona miniatura - ramka i znacznik od razu po kliknieciu (Gradio nie
   przerysowuje galerii w chwili klikniecia, wiec znaczek nakladamy w przegladarce) */
#hist-gallery .thumbnail-lg { position: relative; }
#hist-gallery .thumbnail-lg.hist-sel { outline: 3px solid var(--c-accent) !important; outline-offset: -3px; }
#hist-gallery .thumbnail-lg.hist-sel::after { content: ""; position: absolute; top: 8px; right: 8px;
  width: 24px; height: 24px; border-radius: 999px; z-index: 5; background: var(--c-accent)
  url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m5 12.5 4.5 4.5L19 7.5'/%3E%3C/svg%3E")
  center / 14px no-repeat; box-shadow: 0 1px 4px rgba(0, 0, 0, .3); }

/* --- waskie okno / telefon --- */
@media (max-width: 1100px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 768px) {  /* telefon: miejsce na przycisk panelu nad tytulem */
  .kpis { grid-template-columns: minmax(0, 1fr); }
  .page-head { padding-top: 34px; }
}
"""

# Straznik starej karty. Gradio wysyla zdarzenia po numerze, a aplikacja po restarcie moze miec
# inny uklad - zdarzenie ze starej karty trafiloby do innej funkcji (w logu: "suggest ... needed
# 2, got 0"; rownie dobrze mogla to byc pauza). Gradio sam przeladowuje karte tylko, gdy jakies
# zdarzenie nie przejdzie w trakcie restartu - karta w tle moze restart przespac. Dlatego przed
# kazdym zdarzeniem karta pyta serwer o app_id (nowy proces = nowe): inne - zdarzenie nie
# wychodzi, karta sie przeladowuje. Gdy sprawdzic sie nie da (inna wersja Gradio) - nie blokuje.
STALE_TAB_GUARD = """
  const cfg = window.gradio_config || {};
  const mine = cfg.app_id;
  const send = window.fetch.bind(window);
  const idUrl = location.origin + (cfg.api_prefix || '/gradio_api') + '/app_id';
  const serverId = () => send(idUrl, {cache: 'no-store'})
    .then(r => (r.ok ? r.json().then(j => j.app_id) : mine))
    .catch(() => null);  // serwer nie odpowiada - zdarzenie i tak nie przejdzie
  let stale = false;
  if (mine !== undefined && mine !== null) serverId().then(first => {
    if (first !== mine) return;  // inne juz przy starcie - bez straznika (zadnej petli)
    window.fetch = async (input, init) => {
      const url = String(input instanceof Request ? input.url : input);
      if (url.includes('/queue/join') || url.includes('/run/')) {
        const now = stale ? null : await serverId();
        if (stale || (now !== null && now !== mine)) {
          if (!stale) { stale = true; location.reload(); }
          return new Promise(() => {});  // zdarzenie ze starej karty nie wychodzi
        }
      }
      return send(input, init);
    };
  });
"""
# Zawsze jasny motyw: Gradio dodaje "dark" do body przy ciemnym motywie systemu (tez pozniej,
# gdy system go zmieni) - zdejmujemy ja od razu. Menu: przelacza strone klikiem w ukryty
# przycisk zakladki (od razu w przegladarce; Tab.select dziala jak przy zwyklym kliknieciu).
SHELL = """
  const light = () => { document.body.classList.remove('dark');
    document.documentElement.classList.remove('dark'); };
  light();
  new MutationObserver(light).observe(document.body, {attributes: true, attributeFilter: ['class']});
  // telefon: panel zaslania caly ekran - zamykamy go na starcie i po wyborze strony
  const closeOnPhone = () => {
    if (window.innerWidth <= 768 && document.querySelector('#sidebar.open'))
      document.querySelector('#sidebar .toggle-button')?.click();
  };
  setTimeout(closeOnPhone, 300);
  window.aisGo = (page) => {
    const b = document.querySelector(`#pages [role=tablist] button[data-tab-id="${page}"]`);
    if (b && b.getAttribute('aria-selected') !== 'true') b.click();
    requestAnimationFrame(() => window.scrollTo({top: 0, left: 0, behavior: 'instant'}));
    closeOnPhone();
  };
"""
JS = """() => {
""" + SHELL + """
  const sel = '#hist-gallery .thumbnail-lg';
  const clear = () => document.querySelectorAll(sel + '.hist-sel')
    .forEach(t => t.classList.remove('hist-sel'));
  document.addEventListener('click', (e) => {
    const t = e.target.closest(sel);
    if (t) {
      const multi = document.querySelector('#hist-multi input');
      if (multi && multi.checked) { t.classList.toggle('hist-sel'); }
      else { clear(); t.classList.add('hist-sel'); }
      return;
    }
    if (e.target.closest('#hist-none, #hist-refresh, #hist-filter, #hist-multi')) setTimeout(clear, 50);
    if (e.target.closest('#hist-all')) setTimeout(() => document.querySelectorAll(sel)
      .forEach(x => x.classList.add('hist-sel')), 400);
    if (e.target.closest('#hist-del')) setTimeout(() => {
      if (document.body.innerText.includes('Moved to Recycle Bin:')) clear(); }, 2500);
  }, true);
""" + STALE_TAB_GUARD + "}"
# zmiana pozycji menu -> strona (tylko w przegladarce, bez serwera)
NAV_JS = "(page) => { if (window.aisGo) window.aisGo(page); }"


def nav_css(items: Sequence[tuple]) -> str:
    """CSS menu bocznego: ikona i naglowek grupy nad pozycja (items: etykieta, strona, grupa;
    grupa "-" = sama kreska oddzielajaca). Pozycje rozpoznaje po etykiecie (data-testid)."""
    out = []
    for label, page, group in items:
        sel = f'#nav label[data-testid="{label}-radio-label"]'
        if page in ICONS:
            out.append(f'{sel} {{ --icon: url("{_svg_url(ICONS[page])}"); }}')
        if group == "-":
            out.append(f"{sel} {{ margin-top: 17px; }} {sel}::before {{ content: ''; top: -9px; "
                       "right: 10px; border-top: 1px solid var(--c-border); }")
        elif group:
            out.append(f'{sel} {{ margin-top: 30px; }} {sel}::before {{ content: "{group}"; }}')
    return "\n".join(out)


# ----------------------------- elementy stron -----------------------------

def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


LEGACY_NOTICE = ("Unverified legacy files are preserved on disk and hidden until their SFW review.")


def head_html(title: str, subtitle: str = "") -> str:
    """Naglowek strony. subtitle to gotowy HTML (tresc od uzytkownika - przez esc())."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    return f"<div class='page-head'><h1>{esc(title)}</h1>{sub}</div>"


def page_head(title: str, subtitle: str = "") -> gr.HTML:
    return gr.HTML(head_html(title, subtitle), padding=False)


def section(title: str, subtitle: str = "") -> gr.HTML:
    sub = f"<p class='section-sub'>{subtitle}</p>" if subtitle else ""
    return gr.HTML(f"<div><h3 class='section-title'>{esc(title)}</h3>{sub}</div>", padding=False)


def badge(text: str, kind: str = "") -> str:
    return f"<span class='badge {kind}'>{esc(text)}</span>"


def bar(fraction: float) -> str:
    pct = max(0, min(100, int(round(fraction * 100))))
    return f"<div class='bar'><div style='width:{max(pct, 2)}%'></div></div>"


def table(headers: Iterable[str], rows: List[List[str]], empty: str = "") -> str:
    """Tabela HTML. Komorki to gotowy HTML (tekst od uzytkownika - przez esc())."""
    if not rows:
        return f"<div class='empty'>{empty}</div>" if empty else ""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f"<div class='tbl-wrap'><table class='tbl'><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def kpis(cards: List[dict]) -> str:
    """Rzad wskaznikow: label, value, opcjonalnie sub, small (dluzszy tekst), progress (0-1)."""
    out = []
    for c in cards:
        value_cls = "k-value small" if c.get("small") else "k-value"
        sub = f"<div class='k-sub'>{c['sub']}</div>" if c.get("sub") else ""
        prog = bar(c["progress"]) if c.get("progress") is not None else ""
        out.append(f"<div class='kpi'><div class='k-label'>{esc(c['label'])}</div>"
                   f"<div class='{value_cls}'>{c['value']}</div>{sub}{prog}</div>")
    return f"<div class='kpis'>{''.join(out)}</div>"


# Komunikaty pod przyciskami: znak na poczatku (z modulow aplikacji) -> kolorowa kropka
_MARKS = (("✅", "ok"), ("➕", "ok"), ("✨", "ok"), ("📦", "ok"), ("🗑️", "ok"), ("🗑", "ok"),
          ("❌", "err"), ("⛔", "err"), ("⚠️", "warn"), ("⚠", "warn"), ("⏹", "info"),
          ("📂", "info"), ("ℹ️", "info"))


def note(text: Optional[str]) -> str:
    """Komunikat statusu (Markdown) w spokojnej formie: wiodacy znak ✅/❌/⚠… zamienia na
    kolorowa kropke. Bez znaku - tekst bez zmian."""
    if not text:
        return text or ""
    body = text.lstrip()
    for mark, kind in _MARKS:
        if body.startswith(mark):
            return f"<span class='st st-{kind}'></span>{body[len(mark):].lstrip()}"
    return text


STYLE = {"theme": THEME, "css": CSS, "js": JS}
