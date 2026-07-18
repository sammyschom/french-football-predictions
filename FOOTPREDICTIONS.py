import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.stats import poisson
from itertools import combinations

df = pd.read_csv("DataFoot.csv",sep=";")

# unique list of all teams
toutes_equipes = pd.unique(df[['equipe1', 'equipe2']].values.flatten())

# dictionary mapping each team to all games they played in 
equipe_matches_dict = {
    equipe: df [(df['equipe1'] == equipe) | (df['equipe2'] == equipe)]
    for equipe in toutes_equipes} 

# initialise empty dicts for avg goals and corresponding uncertainties 
buts_moyens_dict = {}
incertitudes_dict = {}

# for each team calculate average goals per match (lambda)
for equipe, matches in equipe_matches_dict.items():
    buts_totals = (
        matches[matches['equipe1'] == equipe]['but1'].sum() + 
        matches[matches['equipe2'] == equipe]['but2'].sum())
    num_matches = len(matches) 
    buts_moyens = buts_totals / num_matches 
    buts_moyens_dict[equipe] = buts_moyens #add each to the dictionary 
    
    #uncertainty calculation
    incertitude = np.sqrt(buts_moyens/ num_matches)
    incertitudes_dict[equipe] = incertitude
    
# print lambda for each team, and the corresponding uncertainty 
results1 = pd.DataFrame({
    'equipe': toutes_equipes,
    'lambda_estimate': [buts_moyens_dict[equipe] for equipe in toutes_equipes],
    'Incertitude':[incertitudes_dict[equipe] for equipe in toutes_equipes]})
print(results1)    

#------------------------------------------------------------------------------

# display poisson distribution for four random teams :

selected_teams = np.random.choice(toutes_equipes, size = 4, replace = False)
for team in selected_teams:
    team_matches = equipe_matches_dict[team] 
    goals = np.where(team_matches['equipe1'] == team, 
                    team_matches['but1'], 
                    team_matches['but2'])

    # create histogram vs line plot
    plt.figure(figsize=(8, 4))
    max_goal = int(goals.max())
    bins = np.arange(-0.5, max_goal + 1.5, 1)
    plt.hist(goals, bins=bins, edgecolor='black', label='Données réelles')
    lambda_est = buts_moyens_dict[team]
    k = np.arange(0, max_goal + 1)
    poisson_prob = (np.exp(-lambda_est) * (lambda_est**k)) / factorial(k) # poisson formula
    expected_counts = poisson_prob * len(goals)
    plt.plot(k, expected_counts, 'ro--', linewidth=2, 
             markersize=5, label=f'Modèle de Poisson (λ = {lambda_est:.2f})')
    plt.title(f'But distribution vs modèle de Poisson: {team}\n'
             f'λ estimate: {lambda_est:.2f} ± {incertitudes_dict[team]:.2f}')
    plt.xlabel('Buts par match ')
    plt.ylabel('Nombre de matchs')
    plt.legend()
plt.show()    

#------------------------------------------------------------------------------

# using the poisson distribution to predict scoreline matrix, returns combined prob of win, draw, loss
# note: the following assumes the two teams goals independent which is not true, but a valid approximation

def calculer_probabilites_poisson(lambda1, lambda2, max_buts=10):
    #initialise probs at zero
    prob_equipe1_win = 0
    prob_nul = 0
    prob_equipe2_win = 0
    
    for k in range(max_buts):
        for m in range(max_buts):
            prob = poisson.pmf(k, lambda1) * poisson.pmf(m, lambda2)
            if k > m:
                prob_equipe1_win += prob
            elif k == m:
                prob_nul += prob
            else:
                prob_equipe2_win += prob      
    return prob_equipe1_win, prob_nul, prob_equipe2_win
  
all_pairs = list(combinations(toutes_equipes, 2))

all_results = []

for team1, team2 in all_pairs:
    lambda1 = buts_moyens_dict[team1]
    lambda2 = buts_moyens_dict[team2]
    
    p_win, p_draw, p_lose = calculer_probabilites_poisson(lambda1, lambda2)
    
    all_results.append({
        'equipe1': team1,
        'equipe2': team2,
        'but1 predit': p_win,
        'null predit': p_draw,
        'but2 predit': p_lose
    })

# Convert the list to a DataFrame and print 
results2 = pd.DataFrame(all_results)
print(results2)

#------------------------------------------------------------------------------

# List of teams that DID reach the round of 16 (given)
round_of_16_teams = [
    'Ajaccio', 'Auxerre', 'Bordeaux', 'Brest', 'Clermont', 'Dijon', 'Lille', 'Metz',
    'Montpellier', 'Nantes', 'Nice', 'OlympiqueLyonnais', 'OlympiqueMarseille',
    'PSG', 'Rennes', 'Strasbourg'
]
# generates random goal count distributed about that team's lambda 
def match_simulation(team1, team2, lambda_dict):
    lambda1 = lambda_dict[team1]
    lambda2 = lambda_dict[team2]
    
    # draws get re-drawn (ET + pens for knock-out)
    while True:
        goals1 = np.random.poisson(lambda1)
        goals2 = np.random.poisson(lambda2)
        
        if goals1 != goals2:
            return (team1, team2) if goals1 > goals2 else (team2, team1)


def tournament_simulation(teams, lambda_dict):
    np.random.shuffle(teams) # random brackets 
    current_round = teams.copy()
    stage_history = {team: [] for team in teams}  # tracks which stages each team reaches
    
    round_names = ["R16", "QF", "SF", "F"]  # Stage labels
    
    stage = 0
    while len(current_round) > 1:
        # record that these teams reached current stage
        for team in current_round:
            stage_history[team].append(round_names[stage])
        
        # play matches (same as before)
        next_round = []
        for i in range(0, len(current_round), 2):
            winner, _ = match_simulation(current_round[i], current_round[i+1], lambda_dict)
            next_round.append(winner)
        
        current_round = next_round
        stage += 1
    
    # Champion gets "W" (winner)
    champion = current_round[0]
    stage_history[champion].append("W")
    return champion, stage_history  # Now returns both winner and stage data

# adjust number of simulations here 
num_simulations = 1000
stage_counts = {team: {"R16":0, "QF":0, "SF":0, "F":0, "W":0} for team in round_of_16_teams}

# Run the tournament num_simulations times, keeps a tally of the winners 

for _ in range(num_simulations):
    winner, stages = tournament_simulation(round_of_16_teams.copy(), buts_moyens_dict)
    for team, reached in stages.items():
        for stage in reached:
            stage_counts[team][stage] += 1

# note: all teams should have R16 as 100.  proof of concept 

results_df = pd.DataFrame([
    {
        "Team": team,
        "R16": stage_counts[team]["R16"] / num_simulations * 100,
        "QF": stage_counts[team]["QF"] / num_simulations * 100,
        "SF": stage_counts[team]["SF"] / num_simulations * 100,
        "F": stage_counts[team]["F"] / num_simulations * 100,
        "W": stage_counts[team]["W"] / num_simulations * 100,
    }
    for team in round_of_16_teams
]).sort_values("W", ascending=False)

print(results_df)

plt.bar(results_df['Team'], results_df['W'])
plt.title('Probability of winning the tournament (%)')
plt.xlabel('Team')
plt.ylabel('Win probability (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
