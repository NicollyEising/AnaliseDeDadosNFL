import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# 1. Defina o diretório base dos CSVs
# -----------------------------
base_path = r'.\nfl-big-data-bowl-2025'

# -----------------------------
# 2. Leitura dos arquivos principais
# -----------------------------
games_df = pd.read_csv(os.path.join(base_path, 'games.csv'))
plays_df = pd.read_csv(os.path.join(base_path, 'plays.csv'))
players_df = pd.read_csv(os.path.join(base_path, 'players.csv'))

# -----------------------------
# 3. Carregue dinamicamente todos os arquivos de tracking por semana
# -----------------------------
tracking_files = glob.glob(os.path.join(base_path, 'tracking_week_*.csv'))
tracking_list = []
for fpath in tracking_files:
    df = pd.read_csv(fpath)
    week_num = int(os.path.basename(fpath).split('_')[2].split('.')[0])
    df['week'] = week_num
    tracking_list.append(df)
tracking_df = pd.concat(tracking_list, ignore_index=True)

# -----------------------------
# 4. Total de pontos por jogo
# -----------------------------
for col in ['homeFinalScore', 'visitorFinalScore']:
    if col not in games_df.columns:
        raise KeyError(f"Coluna esperada ausente em games.csv: {col}")

games_df['total_points'] = games_df['homeFinalScore'] + games_df['visitorFinalScore']
print("=== Total de pontos por jogo ===")
print(games_df[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr', 
                'homeFinalScore', 'visitorFinalScore', 'total_points']].head(), end='\n\n')

# -----------------------------
# 5. Percentual de pontos após o intervalo
# -----------------------------
if {'homeScoreSecondHalf', 'visitorScoreSecondHalf'}.issubset(games_df.columns):
    games_df['points_2nd_half'] = games_df['homeScoreSecondHalf'] + games_df['visitorScoreSecondHalf']
    games_df['pct_2nd_half'] = games_df['points_2nd_half'] / games_df['total_points']
    media_pct = games_df['pct_2nd_half'].mean()
    print(f"Percentual médio de pontos marcados no 2º tempo (segundo tempo): {media_pct:.2%}\n")
else:
    print("Aviso: colunas 'homeScoreSecondHalf' e/ou 'visitorScoreSecondHalf' não encontradas – pulando cálculo desse percentual.\n")

# -----------------------------
# 6. Média de pontos por jogo por equipe
# -----------------------------
home = games_df[['gameId', 'homeTeamAbbr', 'homeFinalScore']].rename(columns={
    'homeTeamAbbr': 'team', 'homeFinalScore': 'points'
})
visitor = games_df[['gameId', 'visitorTeamAbbr', 'visitorFinalScore']].rename(columns={
    'visitorTeamAbbr': 'team', 'visitorFinalScore': 'points'
})
teams_points = pd.concat([home, visitor], ignore_index=True)

mean_points_per_team = teams_points.groupby('team')['points'].mean().reset_index().sort_values(by='points', ascending=False)
print("=== Média de pontos por jogo por equipe ===")
print(mean_points_per_team, end='\n\n')

# -----------------------------
# 7. Distribuição de pontos marcados por jogo
# -----------------------------
media = games_df['total_points'].mean()
mediana = games_df['total_points'].median()
moda = games_df['total_points'].mode()[0]
desvio_padrao = games_df['total_points'].std()

print("=== Estatísticas dos pontos por jogo ===")
print(f"Média: {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Moda: {moda}")
print(f"Desvio padrão: {desvio_padrao:.2f}")

plt.figure(figsize=(8, 5))
plt.hist(games_df['total_points'], bins=20, color='skyblue', edgecolor='black')
plt.axvline(media, color='red', linestyle='dashed', linewidth=1, label='Média')
plt.axvline(mediana, color='green', linestyle='dashed', linewidth=1, label='Mediana')
plt.title('Distribuição de Pontos por Jogo')
plt.xlabel('Total de pontos')
plt.ylabel('Frequência')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 8. Estatísticas de jogadores por semana
# -----------------------------
def estatisticas_jogador_semana(tracking_df, players_df, jogador, semana):
    jogador_info = players_df[players_df['displayName'] == jogador]
    if jogador_info.empty:
        print(f"Jogador '{jogador}' não encontrado.")
        return

    player_id = jogador_info['nflId'].values[0]
    dados = tracking_df[(tracking_df['nflId'] == player_id) & (tracking_df['week'] == semana)]

    if dados.empty:
        print(f"Nenhum dado de tracking encontrado para {jogador} na semana {semana}.")
        return

    # Distância percorrida por jogada (soma das distâncias entre frames consecutivos)
    distancia = dados.groupby('playId').apply(
        lambda x: np.sum(np.sqrt(np.diff(x['x'])**2 + np.diff(x['y'])**2))
    )
    velocidade_media = dados['s'].mean()
    aceleracao_media = dados['a'].mean()
    posicao_media = dados[['x', 'y']].mean()

    print(f"\n=== Estatísticas de {jogador} na semana {semana} ===")
    print(f"Distância média percorrida por jogada: {distancia.mean():.2f} yards")
    print(f"Velocidade média: {velocidade_media:.2f} yards/s")
    print(f"Aceleração média: {aceleracao_media:.2f} yards/s²")
    print(f"Posição média: X={posicao_media['x']:.2f}, Y={posicao_media['y']:.2f}")

# Exemplo de uso
estatisticas_jogador_semana(tracking_df, players_df, 'Tom Brady', 1)

# -----------------------------
# 9. Estatísticas de equipes por semana
# -----------------------------
teams_points['week'] = teams_points['gameId'].map(games_df.set_index('gameId')['week'])

teams_agg = teams_points.groupby(['team', 'week'])['points'].mean().reset_index(name='avg_points_scored')

home_def = games_df[['gameId', 'homeTeamAbbr', 'visitorFinalScore', 'week']].rename(columns={
    'homeTeamAbbr': 'team', 'visitorFinalScore': 'points_against'
})
visitor_def = games_df[['gameId', 'visitorTeamAbbr', 'homeFinalScore', 'week']].rename(columns={
    'visitorTeamAbbr': 'team', 'homeFinalScore': 'points_against'
})
points_against = pd.concat([home_def, visitor_def])

points_against_mean = points_against.groupby(['team', 'week'])['points_against'].mean().reset_index()

team_week_stats = pd.merge(teams_agg, points_against_mean, on=['team', 'week'])
print("\n=== Estatísticas de equipes por semana ===")
print(team_week_stats.head(), end='\n\n')

# -----------------------------
# 10. Análise de desempenho de jogadores por condição (casa/fora)
# -----------------------------
def desempenho_jogador_condicao(tracking_df, players_df, jogador, condicao):
    jogador_info = players_df[players_df['displayName'] == jogador]
    if jogador_info.empty:
        print(f"Jogador '{jogador}' não encontrado.")
        return

    player_id = jogador_info['nflId'].values[0]
    jogos = games_df[games_df['homeTeamAbbr'] == jogador_info['teamAbbr'].values[0] if condicao == 'casa' else games_df['visitorTeamAbbr'] == jogador_info['teamAbbr'].values[0]]
    
    tracking_jogador = tracking_df[(tracking_df['nflId'] == player_id) & (tracking_df['gameId'].isin(jogos['gameId']))]

    if tracking_jogador.empty:
        print(f"Nenhum dado de tracking encontrado para {jogador} na condição '{condicao}'.")
        return

    distancia = tracking_jogador.groupby('playId').apply(
        lambda x: np.sum(np.sqrt(np.diff(x['x'])**2 + np.diff(x['y'])**2))
    )
    velocidade_media = tracking_jogador['s'].mean()
    aceleracao_media = tracking_jogador['a'].mean()

    print(f"\n=== Desempenho de {jogador} ({condicao}) ===")
    print(f"Distância média percorrida por jogada: {distancia.mean():.2f} yards")
    print(f"Velocidade média: {velocidade_media:.2f} yards/s")
    print(f"Aceleração média: {aceleracao_media:.2f} yards/s²")

# -----------------------------
# 11. Análise de tendências ao longo das semanas
# -----------------------------
media_por_semana = games_df.groupby('week')['total_points'].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.plot(media_por_semana['week'], media_por_semana['total_points'], marker='o')
plt.title('Tendência: Média de pontos por jogo por semana')
plt.xlabel('Semana')
plt.ylabel('Média de pontos')
plt.grid(True)
plt.show()


# -----------------------------
# 12. Exemplo opcional: heatmap posição média jogador na semana (se desejar)
# -----------------------------
def posicao_media_jogador(tracking_df, players_df, jogador, semana):
    jogador_info = players_df[players_df['displayName'] == jogador]
    if jogador_info.empty:
        print(f"Jogador '{jogador}' não encontrado.")
        return None

    player_id = jogador_info['nflId'].values[0]
    tracking_jogador = tracking_df[(tracking_df['nflId'] == player_id) & (tracking_df['week'] == semana)]

    if tracking_jogador.empty:
        print(f"Nenhum dado de tracking encontrado para {jogador} na semana {semana}.")
        return None

    posicao_media = tracking_jogador[['x', 'y']].mean()
    print(f"Posição média de {jogador} na semana {semana}: X={posicao_media['x']:.2f}, Y={posicao_media['y']:.2f}")

    return tracking_jogador[['x', 'y']]

# Uso do heatmap
jogador = 'Tom Brady'
semana = 1
jogador_df = posicao_media_jogador(tracking_df, players_df, jogador, semana)

if jogador_df is not None:
    try:
        import seaborn as sns
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=jogador_df, x='x', y='y', fill=True, cmap='Reds')
        plt.title(f'Heatmap de posição: {jogador} (semana {semana})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Para gerar o heatmap, instale o seaborn (pip install seaborn).")
