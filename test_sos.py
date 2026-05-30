from sos_engine import SOSEngine

engine = SOSEngine('data/sos_2026.csv')

print('📊 Demo de Ajustes SOS:')
print('=' * 60)
for team in ['Argentina', 'Portugal', 'Curacao', 'New Zealand']:
    form_adj = engine.calculate_form_adjustment(team)
    sched_adj = engine.calculate_schedule_context_adjustment(team)
    total = form_adj + sched_adj
    print(f'{team:20s}: Forma {form_adj:+.1%} | Calendario {sched_adj:+.1%} | Total {total:+.1%}')
