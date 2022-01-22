import pandas as pd
import aiohttp
import asyncio
import nest_asyncio
import json

from datetime import datetime
from tqdm import trange, tqdm
from understat import Understat
nest_asyncio.apply()

def routine(functionality):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(functionality)

async def save_team_stats(teams, season):
    unique_teams = list(set(teams))
    for i, team in zip(trange(len(unique_teams)), unique_teams):
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            results = await understat.get_team_results(team, season)
        results = pd.DataFrame(results)
        results.to_csv('data/understat/match_stats/{}/{}.csv'.format(season, team), index=False)
        
async def save_player_stats(player_ids, season):
    unique_players = list(set(player_ids))
    for i, player in zip(trange(len(unique_players)), unique_players):
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            matches = await understat.get_player_matches(player, season=str(season))
        matches = pd.DataFrame(matches)
        matches.to_csv('data/understat/player_stats/{}/{}.csv'.format(season, player), index=False)
        
def get_team_stats(team_name, season, date=None, opponent=False):

    results = pd.read_csv('data/understat/match_stats/{}/{}.csv'.format(season, team_name), parse_dates=['datetime'])
    
    if date:
        date = datetime.strptime(date, '%Y-%m-%d').date()
    
    results['date'] = results['datetime'].dt.date
    results = results[results['date'] < date] if date else results
    
    results['GH']  = results['goals'].apply(lambda x: float(json.loads(x.replace('\'', '"'))['h']))
    results['GA']  = results['goals'].apply(lambda x: float(json.loads(x.replace('\'', '"'))['a']))
    results['xGH'] = results['xG'].apply(lambda x: float(json.loads(x.replace('\'', '"'))['h']))
    results['xGA'] = results['xG'].apply(lambda x: float(json.loads(x.replace('\'', '"'))['a']))
    
    results['GT']   = results['GH']
    results['GCT']  = results['GA']
    results['xGT']  = results['xGH']
    results['xGCT'] = results['xGA']
    
    results.loc[results['side']=='a', 'GT']   = results['GA']
    results.loc[results['side']=='a', 'GCT']  = results['GH']
    results.loc[results['side']=='a', 'xGT']  = results['xGA']
    results.loc[results['side']=='a', 'xGCT'] = results['xGH']
    
    average = results[['GT', 'GCT', 'xGT', 'xGCT']].mean()
    
    G90    = average['GT']
    GC90   = average['GCT']
    xG90T  = average['xGT']
    xGC90T = average['xGCT']
    
    if opponent:
        columns = ['G90OT', 'GC90OT', 'xG90OT', 'xGC90OT']
    else:
        columns = ['G90T', 'GC90T', 'xG90T', 'xGC90T']
        
    return pd.DataFrame(data=[[G90, GC90, xG90T, xGC90T]], columns=columns)

def get_player_stats(player_id, season, date=None):
    
    matches = pd.read_csv('data/understat/player_stats/{}/{}.csv'.format(season, player_id), parse_dates=['date'])
    
    if date:
        date = datetime.strptime(date, '%Y-%m-%d').date()
    
    matches['date'] = matches['date'].dt.date
    matches = matches[matches['date'] < date] if date else matches

    matches['time'] = matches['time'].astype(float)
    matches = matches[matches['time'] >= 15]
    
    matches = matches[[
            'goals',
            'assists',
            'shots',
            'xG',
            'xA',
            'time',
        ]]
    
    time = matches['time'].sum()
    
    columns = ['G90P', 'A90P', 's90P', 'xG90P', 'xA90P', 'time']
    
    if time > 0:
        
        G90P  = matches['goals'].astype(float).sum() * 90 / time
        A90P  = matches['assists'].astype(float).sum() * 90 / time
        s90P  = matches['shots'].astype(float).sum() * 90 / time
        xG90P = matches['xG'].astype(float).sum() * 90 / time
        xA90P = matches['xA'].astype(float).sum() * 90 / time
        
        return pd.DataFrame(data=[[G90P, A90P, s90P, xG90P, xA90P, time]], columns=columns)
    
    else:
        
        return pd.DataFrame(data=[[0., 0., 0., 0., 0., 0.]], columns=columns)
    
def get_features(seasons):
    for season in seasons:
        print('Creating features for season {} ...'.format(season))
        population = pd.read_csv('data/outputs/population/{}.csv'.format(season))

        home_team_list     = population['home_team'].to_list()
        away_team_list     = population['away_team'].to_list()
        is_home_list       = population['is_home'].to_list()
        player_ids         = population['player_id'].tolist()
        dates              = population['date'].tolist()

        friendly_teams     = [x[1-x[2]] for x in zip(home_team_list, away_team_list, is_home_list)]
        enemy_teams        = [x[x[2]] for x in zip(home_team_list, away_team_list, is_home_list)]

        population['team'] = friendly_teams
        teams              = population['team'].tolist()

        routine(save_team_stats(teams, season))
        routine(save_player_stats(player_ids, season))

        players  = None
        friendly = None
        enemy    = None

        for i, player_id, friendly_team, enemy_team, date in zip(trange(len(player_ids)), player_ids, friendly_teams, enemy_teams, dates):
            
            player_stats        = get_player_stats(player_id, season, date)
            friendly_team_stats = get_team_stats(friendly_team, season, date)
            enemy_team_stats    = get_team_stats(enemy_team, season, date, opponent=True)

            players             = players.append(player_stats, ignore_index=True) if players is not None else player_stats
            friendly            = friendly.append(friendly_team_stats, ignore_index=True) if friendly is not None else friendly_team_stats
            enemy               = enemy.append(enemy_team_stats, ignore_index=True) if enemy is not None else enemy_team_stats

        dataset = population.join([players, friendly, enemy])
        dataset = dataset.fillna(0)
        dataset.to_csv('data/outputs/raw/{}.csv'.format(season), index=False)
        
def get_dataset(seasons):
    print('Merging dataset for each season ...')
    df = None
    for season in seasons:
        to_add = pd.read_csv('data/outputs/raw/{}.csv'.format(season))
        df = df.append(to_add, ignore_index=True) if df is not None else to_add
        df = df[df.position != 'GK']
        df = df[[
            'match_id',
            'date',
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
            'xGC90OT',

            'target'
        ]]       
        df.to_csv('data/outputs/dataset_{}_{}.csv'.format(seasons[0], seasons[-1]), index=False)