s6lc: скрипты отчёта (после итогового анализа). Порядок: lc_report_tables.py --dir analysis --out analysis/tables.md; lc_reading.py --contrasts analysis/lc_contrasts.csv --out analysis/reading.md; lc_posthoc.py --dir analysis --out analysis/posthoc.md;
lc_figure.py (make_fig1, make_fig2 в python после apply_figure_style); build_s6lc_report.py --dir analysis --fig1 <id> --fig2 <id> --out paper4_s6lc_ru.md.
Замороженные скрипты обучения и анализа (lc_*.py на сервере) лежат в s6lc_code_phase1.zip.
