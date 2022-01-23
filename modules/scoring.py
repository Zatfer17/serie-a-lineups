import pandas as pd
import numpy as np
import pickle
import aiohttp
import asyncio
import nest_asyncio
import json

from tqdm import trange, tqdm
from datetime import datetime
from understat import Understat
from modules.features import get_team_stats, get_player_stats
nest_asyncio.apply()

def routine(functionality):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(functionality)

async def save_team_fixtures(teams, season):
    unique_teams = list(set(teams))
    for i, team in zip(trange(len(unique_teams)), unique_teams):
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            fixtures = await understat.get_team_fixtures(team_name=team, season=season)
        fixtures = pd.DataFrame(fixtures)
        fixtures.to_csv('data/understat/team_fixtures/{}/{}.csv'.format(season, team), index=False)
        
async def get_team_fixtures(team, season, date=None):

    fixtures = pd.read_csv('data/understat/team_fixtures/{}/{}.csv'.format(season, team), parse_dates=['datetime'])
    
    if date:
        date = datetime.strptime(date, '%Y-%m-%d').date()
    
    fixtures['team']      = team
    fixtures['home_team'] = fixtures['h'].apply(lambda x: json.loads(x.replace('\'', '"'))['title'])
    fixtures['away_team'] = fixtures['a'].apply(lambda x: json.loads(x.replace('\'', '"'))['title'])
    fixtures['season']    = 2021
    fixtures['month']     = fixtures['datetime'].dt.month.astype(int)
    fixtures['day']       = fixtures['datetime'].dt.day.astype(int)
    fixtures['hour']      = fixtures['datetime'].dt.hour.astype(int)
    fixtures['week_day']  = fixtures['datetime'].dt.weekday.astype(int)
    fixtures['date']      = fixtures['datetime'].dt.date
    
    fixtures = fixtures[fixtures['date'] >= date] if date else fixtures[fixtures['date'] >= pd.Timestamp('today').date()]
    
    return fixtures[[
        'home_team',
        'away_team',
        'date',
        'season',
        'month',
        'day',
        'hour',
        'week_day'
    ]].head(1)

async def get_player_id(player):
    
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        players = await understat.get_league_players(league_name='Serie A', season = 2021, player_name=player)
    
    if players:
        players = pd.DataFrame(players)
        players['player_id'] = players['id']
        return players['player_id'].tolist()[0]
    else:
        return None
    
async def get_player_position(player_id):
    try:
        matches = pd.read_csv('data/understat/player_stats/{}/{}.csv'.format(2021, player_id))
        return matches['position'].mode().tolist()[0]
    except FileNotFoundError:
        return 'Sub'

async def get_player_bonus(player_id, season, date=None):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        matches = await understat.get_player_matches(player_id, season=str(season))
        
    matches = pd.DataFrame(matches)
    
    matches['date'] = pd.to_datetime(matches['date'])
    matches['date'] = matches['date'].dt.date

    if date:
        date = datetime.strptime(date, '%Y-%m-%d').date()

    matches = matches[matches['date'] == date] if date else matches.head(1)

    if len(matches) == 1:

        goals   = matches['goals'].astype(float).to_list()[0]
        assists = matches['assists'].astype(float).to_list()[0]
        
        return 3*goals + assists

    else:

        return np.nan

    
    
def get_scoring_set(league_name):
    
    league = pd.read_csv('data/leghe_fantacalcio/leagues/{}.csv'.format(league_name))
    league = league[league['role'] != 'P']
    league = league[['fanta_team', 'role', 'team', 'player']]
    
    fixtures   = None
    player_ids = []
    positions  = []
    teams      = league['team'].tolist()
    players    = league['player'].tolist()
    
    routine(save_team_fixtures(teams, 2021))

    for i, team, player in zip(trange(len(teams)), teams, players):
        
        to_add = routine(get_team_fixtures(team, 2021))
        fixtures = fixtures.append(to_add, ignore_index=True) if fixtures is not None else to_add
        
        to_add = routine(get_player_id(player))
        player_ids.append(to_add)
        
        to_add = routine(get_player_position(to_add))
        positions.append(to_add)
        
    player_ids = pd.DataFrame(data=player_ids, columns=['player_id'])
    positions  = pd.DataFrame(data=positions, columns=['position'])

    league     = league.reset_index(drop=True)
    player_ids = player_ids.reset_index(drop=True)
    fixtures   = fixtures.reset_index(drop=True)
    positions  = positions.reset_index(drop=True)

    league = league.join([player_ids, fixtures, positions])
    league = league.dropna()
    
    league['is_home'] = (league['team'] == league['home_team'])
    league['is_home'] = league['is_home'].astype(int)
    
    return league[[
        'fanta_team',
        'date',
        'role',
        'player_id',
        'player',
        'team',
        'home_team',
        'away_team',
        'is_home',
        'season',
        'month',
        'day',
        'hour',
        'week_day',
        'position'
    ]]

def get_features(scoring_set, league_name):
    
    home_team_list = scoring_set['home_team'].to_list()
    away_team_list = scoring_set['away_team'].to_list()
    is_home_list   = scoring_set['is_home'].to_list()
    player_ids     = scoring_set['player_id'].tolist()
    dates          = scoring_set['date'].tolist()
    teams          = scoring_set['team'].tolist()
    
    friendly_teams = [x[1-x[2]] for x in zip(home_team_list, away_team_list, is_home_list)]
    enemy_teams    = [x[x[2]] for x in zip(home_team_list, away_team_list, is_home_list)]
    
    players  = None
    friendly = None
    enemy    = None
    
    for i, player_id, friendly_team, enemy_team, date in zip(trange(len(player_ids)), player_ids, friendly_teams, enemy_teams, dates):
            
            player_stats        = get_player_stats(player_id, 2021)
            friendly_team_stats = get_team_stats(friendly_team, 2021)
            enemy_team_stats    = get_team_stats(enemy_team, 2021, opponent=True)

            players             = players.append(player_stats, ignore_index=True) if players is not None else player_stats
            friendly            = friendly.append(friendly_team_stats, ignore_index=True) if friendly is not None else friendly_team_stats
            enemy               = enemy.append(enemy_team_stats, ignore_index=True) if enemy is not None else enemy_team_stats
            
    scoring_set = scoring_set.reset_index(drop=True)
    players     = players.reset_index(drop=True)
    friendly    = friendly.reset_index(drop=True)
    enemy       = enemy.reset_index(drop=True)
    
    scoring_set = scoring_set.join([players, friendly, enemy])
    scoring_set = scoring_set.fillna(0)
    
    return scoring_set[[
        'fanta_team',
        'date',
        'role',
        'player_id',
        'player',
        'team',

        'home_team',
        'away_team',
        'is_home',

        'season',
        'month',
        'day',
        'hour',
        'week_day',

        'position',
        'G90P',
        'A90P',
        's90P',
        'xG90P',
        'xA90P',
        'time',

        'G90T',
        'GC90T',
        'xG90T',
        'xGC90T',

        'G90OT',
        'GC90OT',
        'xG90OT',
        'xGC90OT'
    ]]

def score(X, scoring_set, league_name, gameweek, save=True):
    
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    scoring_set['expected_bonus'] = model.predict(X)
    
    final = scoring_set[[
        'fanta_team',
        'role',
        'player',
        'home_team',
        'away_team',
        'date',
        'expected_bonus']]
    
    final = final.sort_values(by=['fanta_team', 'role', 'expected_bonus'], ascending=False)
    
    if save:
        final.to_csv('data/outputs/prediction/{}/gameweek_{}.csv'.format(league_name, gameweek), index=False)
    
    return final

def validate(league_name, gameweek):

    predictions = pd.read_csv('data/outputs/prediction/{}/gameweek_{}.csv'.format(league_name, gameweek))

    players = predictions['player'].to_list()
    dates   = predictions['date'].to_list()

    actual_bonuses = []

    for i, player, date in zip(trange(len(players)), players, dates):
        player_id = routine(get_player_id(player))
        to_add = routine(get_player_bonus(player_id, 2021, date))
        actual_bonuses.append(to_add)

    actual_bonuses = pd.DataFrame(data=actual_bonuses, columns=['true_bonus'])

    predictions    = predictions.reset_index(drop=True)
    actual_bonuses = actual_bonuses.reset_index(drop=True)

    validation = predictions.join(actual_bonuses)
    validation = validation.dropna()

    validation['error']      = validation['expected_bonus'] - validation['true_bonus']
    validation['error_type'] = validation['error'].apply(lambda x: 'overestimated' if x >= 0 else 'underestimated')

    validation.to_csv('data/outputs/validation/{}/gameweek_{}.csv'.format(league_name, gameweek))
    
    return validation