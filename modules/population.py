import pandas as pd
import aiohttp
import asyncio
import nest_asyncio

from tqdm import trange, tqdm
from understat import Understat
nest_asyncio.apply()

def routine(functionality):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(functionality)

async def get_league_matches(season):

    async with aiohttp.ClientSession() as session:
        
        understat = Understat(session)
        matches = await understat.get_league_results(league_name='Serie A', season=season)
        matches = pd.DataFrame(matches)
    
    matches['match_id']  = matches['id']
    matches['home_team'] = matches['h'].apply(lambda x: x.get('title'))
    matches['away_team'] = matches['a'].apply(lambda x: x.get('title'))
    matches['season']    = season
    matches['date']      = pd.to_datetime(matches['datetime'])
    matches['month']     = matches['date'].dt.month
    matches['day']       = matches['date'].dt.day
    matches['hour']      = matches['date'].dt.hour
    matches['week_day']  = matches['date'].dt.weekday  
    matches['date']      = matches['date'].dt.date

    return matches[[
        'match_id',
        'date',
        'home_team',
        'away_team',
        'season',
        'month',
        'day',
        'hour',
        'week_day'
    ]]

async def get_match_players(match_id):

    async with aiohttp.ClientSession() as session:
        
        understat = Understat(session)
        players_by_team = await understat.get_match_players(match_id=match_id)
    
    players = []
    for team in players_by_team:
        for player in players_by_team[team]:
            players.append(players_by_team[team][player])
    players = pd.DataFrame(players)
    
    players['match_id'] = match_id
    players['is_home']  = players['h_a'].apply(lambda x: 1 if x == 'h' else 0)
    players['time']     = players['time'].astype(float)
    players['goals']    = players['goals'].astype(float)
    players['assists']  = players['assists'].astype(float)
    players['target']   = 3*players['goals'] + players['assists']
    
    players = players[players['time'] >= 15]
    
    return players[[
        'match_id',
        'player_id',
        'player',
        'is_home',
        'position',
        'target'
    ]]

def get_population(seasons):
    for season in seasons:
        print('Populating season {}'.format(season))
        df = routine(get_league_matches(season))
        match_ids = list(df['match_id'].unique())
        season_data = None
        for i, match_id in zip(trange(len(match_ids)), match_ids):
            players = routine(get_match_players(match_id))
            to_add = df.merge(players, on='match_id')
            season_data = pd.concat([season_data, to_add], ignore_index=True) if season_data is not None else to_add
        season_data = season_data[[
            'match_id',
            'date',
            'player_id',
            'player',
            'home_team',
            'away_team',
            'is_home',
            'season',
            'month',
            'day',
            'hour',
            'week_day',
            'position',
            'target'
        ]]
        season_data.to_csv('data/outputs/population/{}.csv'.format(season), index=False)