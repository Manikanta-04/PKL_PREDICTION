"""
Create Realistic PKL Dataset
Run this to generate a complete synthetic PKL dataset
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

print("=" * 60)
print("🏆 CREATING REALISTIC PKL DATASET")
print("=" * 60)

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# All PKL teams with their home cities
teams = [
    {'name': 'Patna Pirates', 'city': 'Patna', 'strength': 0.85},
    {'name': 'Bengal Warriors', 'city': 'Kolkata', 'strength': 0.82},
    {'name': 'U Mumba', 'city': 'Mumbai', 'strength': 0.78},
    {'name': 'Jaipur Pink Panthers', 'city': 'Jaipur', 'strength': 0.75},
    {'name': 'Haryana Steelers', 'city': 'Haryana', 'strength': 0.72},
    {'name': 'Tamil Thalaivas', 'city': 'Chennai', 'strength': 0.70},
    {'name': 'Telugu Titans', 'city': 'Hyderabad', 'strength': 0.68},
    {'name': 'Puneri Paltan', 'city': 'Pune', 'strength': 0.73},
    {'name': 'Gujarat Giants', 'city': 'Ahmedabad', 'strength': 0.71},
    {'name': 'Dabang Delhi KC', 'city': 'Delhi', 'strength': 0.77},
    {'name': 'Bengaluru Bulls', 'city': 'Bengaluru', 'strength': 0.69},
    {'name': 'UP Yoddhas', 'city': 'Lucknow', 'strength': 0.66}
]

team_names = [t['name'] for t in teams]
team_strength = {t['name']: t['strength'] for t in teams}

print(f"✅ Loaded {len(teams)} PKL teams")

# Generate standings data for 10 seasons
np.random.seed(42)
standings_data = []

for season in range(1, 11):
    print(f"\n📊 Generating Season {season} standings...")
    
    # Randomize team strengths slightly each season
    season_strengths = {}
    for team in team_names:
        # Add some randomness to team strength each season
        season_strengths[team] = team_strength[team] + np.random.normal(0, 0.05)
        season_strengths[team] = max(0.5, min(0.95, season_strengths[team]))
    
    # Sort teams by strength to determine standings
    sorted_teams = sorted(season_strengths.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (team, strength) in enumerate(sorted_teams, 1):
        # Generate season statistics based on strength
        matches = 22  # Each team plays 22 matches in a season
        win_prob = strength
        wins = int(matches * win_prob * 0.9 + np.random.randint(-2, 3))
        losses = int(matches * (1 - win_prob) * 0.9 + np.random.randint(-2, 3))
        ties = matches - wins - losses
        
        # Ensure non-negative
        wins = max(0, min(matches, wins))
        losses = max(0, min(matches - wins, losses))
        ties = matches - wins - losses
        
        points = wins * 5 + ties * 3  # PKL scoring: 5 for win, 3 for tie
        
        standings_data.append({
            'season': season,
            'team_name': team,
            'rank': rank,
            'matches_played': matches,
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'points': points,
            'strength': strength
        })

# Save standings
standings_df = pd.DataFrame(standings_data)
standings_df.to_csv('data/raw_standings.csv', index=False)
print(f"\n✅ Saved {len(standings_df)} standings records")

# Generate matches data
print("\n📊 Generating match data...")
matches_data = []
venues = ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bengaluru', 'Hyderabad', 
          'Pune', 'Jaipur', 'Ahmedabad', 'Lucknow', 'Patna', 'Haryana']

start_date = datetime(2023, 1, 1)

for season in range(1, 11):
    # Get team strengths for this season
    season_strengths = standings_df[standings_df['season'] == season].set_index('team_name')['strength'].to_dict()
    
    # Each team plays every other team twice (home and away)
    n_matches = len(teams) * (len(teams) - 1)  # 12*11 = 132 matches per season
    
    for match_id in range(n_matches):
        # Select two different teams
        team_a, team_b = np.random.choice(team_names, size=2, replace=False)
        
        # Get team strengths
        strength_a = season_strengths.get(team_a, 0.7)
        strength_b = season_strengths.get(team_b, 0.7)
        
        # Home advantage (if venue matches team city)
        venue = np.random.choice(venues)
        home_advantage = 0.05 if venue in [t['city'] for t in teams if t['name'] == team_a] else 0
        
        # Calculate win probability
        team_a_win_prob = 1 / (1 + np.exp(-(strength_a + home_advantage - strength_b)))
        
        # Determine winner
        if np.random.random() < team_a_win_prob:
            winner = team_a
            team_a_score = np.random.randint(32, 45)
            team_b_score = np.random.randint(25, 38)
        else:
            winner = team_b
            team_a_score = np.random.randint(25, 38)
            team_b_score = np.random.randint(32, 45)
        
        # Generate match date
        match_date = start_date + timedelta(days=match_id)
        
        matches_data.append({
            'season': season,
            'match_id': f'PKL_S{season}_M{match_id+1:03d}',
            'team_a': team_a,
            'team_b': team_b,
            'team_a_score': team_a_score,
            'team_b_score': team_b_score,
            'winner': winner,
            'date': match_date.strftime('%Y-%m-%d'),
            'venue': venue,
            'strength_a': strength_a,
            'strength_b': strength_b
        })

# Save matches
matches_df = pd.DataFrame(matches_data)
matches_df.to_csv('data/raw_matches.csv', index=False)
print(f"✅ Saved {len(matches_df)} match records")

# Create ML dataset
print("\n🔧 Creating ML-ready dataset...")
ml_data = []

for _, match in matches_df.iterrows():
    season = match['season']
    team_a = match['team_a']
    team_b = match['team_b']
    
    # Get previous season stats
    prev_standings = standings_df[(standings_df['season'] == season-1) & 
                                   (standings_df['team_name'].isin([team_a, team_b]))]
    
    team_a_prev = prev_standings[prev_standings['team_name'] == team_a]
    team_b_prev = prev_standings[prev_standings['team_name'] == team_b]
    
    team_a_win_pct_prev = team_a_prev['wins'].values[0] / team_a_prev['matches_played'].values[0] if len(team_a_prev) > 0 else 0.5
    team_b_win_pct_prev = team_b_prev['wins'].values[0] / team_b_prev['matches_played'].values[0] if len(team_b_prev) > 0 else 0.5
    
    team_a_points_prev = team_a_prev['points'].values[0] if len(team_a_prev) > 0 else 30
    team_b_points_prev = team_b_prev['points'].values[0] if len(team_b_prev) > 0 else 30
    
    # Current season form (simplified)
    current_matches = matches_df[(matches_df['season'] == season) & 
                                 ((matches_df['team_a'] == team_a) | (matches_df['team_b'] == team_a))]
    team_a_current_wins = len(current_matches[current_matches['winner'] == team_a])
    
    current_matches_b = matches_df[(matches_df['season'] == season) & 
                                   ((matches_df['team_a'] == team_b) | (matches_df['team_b'] == team_b))]
    team_b_current_wins = len(current_matches_b[current_matches_b['winner'] == team_b])
    
    ml_data.append({
        'season': season,
        'match_id': match['match_id'],
        'team_a': team_a,
        'team_b': team_b,
        'team_a_win_pct_prev': team_a_win_pct_prev,
        'team_b_win_pct_prev': team_b_win_pct_prev,
        'team_a_points_prev': team_a_points_prev,
        'team_b_points_prev': team_b_points_prev,
        'team_a_current_wins': team_a_current_wins,
        'team_b_current_wins': team_b_current_wins,
        'team_a_strength': match['strength_a'],
        'team_b_strength': match['strength_b'],
        'winner': match['winner'],
        'target': 1 if match['winner'] == team_a else 0
    })

# Save ML dataset
ml_dataset = pd.DataFrame(ml_data)
ml_dataset.to_csv('data/pkl_ml_dataset.csv', index=False)
print(f"✅ Saved {len(ml_dataset)} ML-ready records")

# Display statistics
print("\n" + "=" * 60)
print("📊 DATASET STATISTICS")
print("=" * 60)
print(f"\nTotal matches: {len(matches_df)}")
print(f"Total seasons: {matches_df['season'].nunique()}")
print(f"Total teams: {matches_df['team_a'].nunique()}")
print(f"\nMatches per season:")
print(matches_df['season'].value_counts().sort_index())

print("\nSample matches:")
print(matches_df[['season', 'team_a', 'team_b', 'winner', 'team_a_score', 'team_b_score']].head(10))

print("\n" + "=" * 60)
print("✅ DATASET CREATION COMPLETE!")
print("=" * 60)
