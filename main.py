import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


# -----------------------------
# Defina o diretório base dos CSVs
# -----------------------------
base_path = r'.\nfl-big-data-bowl-2025'

# -----------------------------
# Leitura dos arquivos principais
# -----------------------------
games_df = pd.read_csv(os.path.join(base_path, 'games.csv'))
plays_df = pd.read_csv(os.path.join(base_path, 'plays.csv'))
players_df = pd.read_csv(os.path.join(base_path, 'players.csv'))

# -----------------------------
# Carregue dinamicamente todos os arquivos de tracking por semana
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
# Total de pontos por jogo
# -----------------------------
required_columns = ['homeFinalScore', 'visitorFinalScore']
missing_columns = [col for col in required_columns if col not in games_df.columns]

if missing_columns:
    raise KeyError(f"Coluna(s) esperada(s) ausente(s) no arquivo: {', '.join(missing_columns)}")

games_df['total_points'] = games_df['homeFinalScore'] + games_df['visitorFinalScore']

print("=== Total de pontos por jogo ===")
print(
    games_df[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr', 
              'homeFinalScore', 'visitorFinalScore', 'total_points']]
    .sort_values(by='total_points', ascending=False)
    .head(10)
    .to_string(index=False),
    end='\n\n'
)

games_sorted = games_df.sort_values(by='total_points', ascending=False).reset_index(drop=True)

top_games = games_sorted.head(20).copy()
top_games['match'] = top_games['homeTeamAbbr'] + " x " + top_games['visitorTeamAbbr']

plt.figure(figsize=(12, 6))
sns.set(style='whitegrid')
# Correção do warning do Seaborn: usar hue para palette sem legenda
sns.barplot(data=top_games, y='match', x='total_points', hue='match', palette='rocket', dodge=False, legend=False)

plt.title('Top 20 Jogos com Maior Total de Pontos', fontsize=14, weight='bold')
plt.xlabel('Total de Pontos', fontsize=12)
plt.ylabel('Partida', fontsize=12)
plt.tight_layout()
plt.show()

# -----------------------------
# Percentual de pontos após o intervalo
# -----------------------------
if {'homeScoreSecondHalf', 'visitorScoreSecondHalf'}.issubset(games_df.columns):
    games_df['points_2nd_half'] = games_df['homeScoreSecondHalf'] + games_df['visitorScoreSecondHalf']
    games_df['pct_2nd_half'] = games_df['points_2nd_half'] / games_df['total_points']

    media_pct = games_df['pct_2nd_half'].mean()
    print(f"Percentual médio de pontos marcados no 2º tempo: {media_pct:.2%}\n")

    plt.figure(figsize=(10, 5))
    sns.set(style='whitegrid')
    sns.histplot(games_df['pct_2nd_half'], bins=20, kde=True, color='orange', edgecolor='black')

    plt.axvline(media_pct, color='red', linestyle='--', linewidth=2, label=f'Média: {media_pct:.2%}')
    plt.title('Distribuição do Percentual de Pontos no 2º Tempo por Jogo', fontsize=14, weight='bold')
    plt.xlabel('Percentual de Pontos no 2º Tempo')
    plt.ylabel('Frequência')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Aviso: colunas 'homeScoreSecondHalf' e/ou 'visitorScoreSecondHalf' não encontradas – pulando cálculo desse percentual.\n")


# -----------------------------
# Média de pontos por jogo por equipe
# -----------------------------
home = games_df[['gameId', 'homeTeamAbbr', 'homeFinalScore']].rename(columns={
    'homeTeamAbbr': 'team', 'homeFinalScore': 'points'
})
visitor = games_df[['gameId', 'visitorTeamAbbr', 'visitorFinalScore']].rename(columns={
    'visitorTeamAbbr': 'team', 'visitorFinalScore': 'points'
})
teams_points = pd.concat([home, visitor], ignore_index=True)

mean_points_per_team = (
    teams_points
    .groupby('team')['points']
    .mean()
    .reset_index()
    .sort_values(by='points', ascending=False)
    .round(2)
)


# -----------------------------
# Estatísticas de jogadores por semana
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

    # Correção do warning do groupby.apply, desabilitando group_keys para excluir colunas agrupamento
    distancia = dados.groupby('playId', group_keys=False).apply(
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

    plt.figure(figsize=(12, 5))
    sns.histplot(distancia, bins=20, kde=True, color='lightgreen')
    plt.title(f'Distribuição da distância por jogada - {jogador} (Semana {semana})')
    plt.xlabel('Distância (yards)')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Exemplo de uso
estatisticas_jogador_semana(tracking_df, players_df, 'Tom Brady', 1)


# -----------------------------
# Estatísticas de equipes por semana
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
# Análise de desempenho de jogadores por condição (casa/fora)
# -----------------------------
def desempenho_jogador_condicao(tracking_df, players_df, jogador, condicao):
    jogador_info = players_df[players_df['displayName'] == jogador]
    if jogador_info.empty:
        print(f"Jogador '{jogador}' não encontrado.")
        return

    player_id = jogador_info['nflId'].values[0]
    team_abbr = jogador_info['teamAbbr'].values[0]
    if condicao == 'casa':
        jogos = games_df[games_df['homeTeamAbbr'] == team_abbr]
    elif condicao == 'fora':
        jogos = games_df[games_df['visitorTeamAbbr'] == team_abbr]
    else:
        print(f"Condição '{condicao}' inválida. Use 'casa' ou 'fora'.")
        return
    
    tracking_jogador = tracking_df[(tracking_df['nflId'] == player_id) & (tracking_df['gameId'].isin(jogos['gameId']))]

    if tracking_jogador.empty:
        print(f"Nenhum dado de tracking encontrado para {jogador} na condição '{condicao}'.")
        return

    distancia = tracking_jogador.groupby('playId', group_keys=False).apply(
        lambda x: np.sum(np.sqrt(np.diff(x['x'])**2 + np.diff(x['y'])**2))
    )
    velocidade_media = tracking_jogador['s'].mean()
    aceleracao_media = tracking_jogador['a'].mean()

    print(f"\n=== Desempenho de {jogador} ({condicao}) ===")
    print(f"Distância média percorrida por jogada: {distancia.mean():.2f} yards")
    print(f"Velocidade média: {velocidade_media:.2f} yards/s")
    print(f"Aceleração média: {aceleracao_media:.2f} yards/s²")

    plt.figure(figsize=(12, 5))
    sns.histplot(distancia, bins=20, kde=True, color='skyblue')
    plt.title(f'Distribuição da distância percorrida por jogada - {jogador} ({condicao})')
    plt.xlabel('Distância (yards)')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Velocidade Média', 'Aceleração Média'], y=[velocidade_media, aceleracao_media], palette='Blues_d')
    plt.title(f'Métricas médias - {jogador} ({condicao})')
    plt.ylabel('Valor (yards/s ou yards/s²)')
    plt.tight_layout()
    plt.show()


# -----------------------------
# Heatmap posição média jogador na semana
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
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=jogador_df, x='x', y='y', fill=True, cmap='Reds')
        plt.title(f'Heatmap de posição: {jogador} (semana {semana})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Para gerar o heatmap, instale o seaborn (pip install seaborn).")


# -----------------------------
# Gráfico: Correlação entre velocidade e aceleração dos jogadores
# -----------------------------
amostra = tracking_df[['s', 'a']].dropna().sample(n=10000, random_state=42)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=amostra, x='s', y='a', alpha=0.4)
plt.title('Correlação entre velocidade e aceleração (amostra)')
plt.xlabel('Velocidade (yards/s)')
plt.ylabel('Aceleração (yards/s²)')
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# Gráfico: Evolução da pontuação média das equipes por semana (linha)
# -----------------------------
team_week_avg = team_week_stats.groupby('week')['avg_points_scored'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.set(style='whitegrid')

sns.lineplot(data=team_week_avg, x='week', y='avg_points_scored', marker='o', color='darkorange', linewidth=2.5)

plt.title('Evolução da Pontuação Média das Equipes por Semana', fontsize=14, weight='bold')
plt.xlabel('Semana', fontsize=12)
plt.ylabel('Média de Pontos por Equipe', fontsize=12)

y_min = team_week_avg['avg_points_scored'].min() * 0.95
y_max = team_week_avg['avg_points_scored'].max() * 1.05
plt.ylim(y_min, y_max)

plt.xticks(team_week_avg['week'])
plt.tight_layout()
plt.show()


# -----------------------------
# Velocidade média dos jogadores por semana
# -----------------------------
velocidade_semana = tracking_df.groupby('week')['s'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=velocidade_semana, x='week', y='s', marker='o', color='teal', linewidth=2.5)

plt.title('Velocidade Média dos Jogadores por Semana', fontsize=14, weight='bold')
plt.xlabel('Semana')
plt.ylabel('Velocidade Média (yards/s)')
plt.grid(True)
plt.tight_layout()
plt.show()
