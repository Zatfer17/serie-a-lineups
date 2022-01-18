import pandas as pd
import aiohttp
import asyncio
from understat import Understat
from datetime import datetime

def routine(functionality):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(functionality)

async def get_matches(seasons):

    print('Getting matches data for each season')

    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for season in seasons:
            print('Getting data for season {}'.format(season))
            results = await understat.get_league_results(
                league_name='Serie A',
                season=season
                )
            data=[]
            for match in results:
                data.append([
                    season,
                    match['id'],
                    match['h']['title'],
                    match['h']['title'],
                    match['a']['title'],
                    match['goals']['h'],
                    match['goals']['a'],
                    match['xG']['h'],
                    match['xG']['a'],
                    90,
                    datetime.strptime(match['datetime'], '%Y-%m-%d %H:%M:%S')
                ])
                data.append([
                    season,
                    match['id'],
                    match['a']['title'],
                    match['h']['title'],
                    match['a']['title'],
                    match['goals']['a'],
                    match['goals']['h'],
                    match['xG']['a'],
                    match['xG']['h'],
                    90,
                    datetime.strptime(match['datetime'], '%Y-%m-%d %H:%M:%S')
                ])
            df = pd.DataFrame(data=data, columns=[
                'season',
                'matchId',
                'team',
                'homeTeam',
                'awayTeam',
                'goalsScored',
                'goalsConceded',
                'xGScored',
                'xGConceded',
                'time',
                'date'
                ])
            df.to_csv('resources/matches/{}/{}.csv'.format(season, season))

def get_grouped_matches(seasons):
    for season in seasons:
        df = pd.read_csv('resources/matches/{}/{}.csv'.format(season, season), index_col=0)
        grouped = df.groupby('team')
        for team, group in grouped:
            group.to_csv('resources/matches/{}/{}.csv'.format(season, team))

def aggregate_previous_matches(seasons):
    for season in seasons:
        df = pd.read_csv('resources/matches/{}/{}.csv'.format(season, season), index_col=0)
        teams = df['team'].unique().tolist()
        for team in teams:
            df = pd.read_csv('resources/matches/{}/{}.csv'.format(season, team), index_col=0)

            goalsScored = 0
            goalsConceded = 0
            xGScored = 0
            xGConceded = 0
            time = 0

            for row in df.iterrows():
                df.loc[row[0], 'goalsScored'] = goalsScored
                df.loc[row[0], 'goalsConceded'] = goalsConceded
                df.loc[row[0], 'xGScored'] = xGScored
                df.loc[row[0], 'xGConceded'] = xGConceded
                df.loc[row[0], 'time'] = time

                goalsScored += row[1]['goalsScored']
                goalsConceded += row[1]['goalsConceded']
                xGScored += row[1]['xGScored']
                xGConceded += row[1]['xGConceded']
                time += row[1]['time']

            df.to_csv('resources/matches/{}/{}.csv'.format(season, team))

def get_complete_matches(seasons):
    complete = None
    for season in seasons:
        df = pd.read_csv('resources/matches/{}/{}.csv'.format(season, season), index_col=0)
        teams = df['team'].unique().tolist()
        semi_complete = None
        for team in teams:
            if semi_complete is None:
                semi_complete = pd.read_csv('resources/matches/{}/{}.csv'.format(season, team), index_col=0)
            else:
                semi_complete = semi_complete.append(pd.read_csv('resources/matches/{}/{}.csv'.format(season, team), index_col=0), ignore_index=True)
        if complete is None:
            complete = semi_complete.merge(semi_complete, on=['season', 'matchId', 'homeTeam', 'awayTeam', 'date'], suffixes=('', 'Opponent'))
        else:
            complete = complete.append(semi_complete.merge(semi_complete, on=['season', 'matchId', 'homeTeam', 'awayTeam', 'date'], suffixes=('', 'Opponent')), ignore_index=True)
    complete = complete[complete.team != complete.teamOpponent]
    complete.to_csv('resources/matches/complete.csv')

async def get_players(seasons):

    print('Getting player matches data for each season')

    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for season in seasons:
            print('Getting data for season {}'.format(season))
            df = pd.read_csv('resources/matches/{}/{}.csv'.format(season, season), index_col=0)
            match_ids = df['matchId'].unique().tolist()
            data = []
            for match_id in match_ids:
                teams = await understat.get_match_players(
                    match_id=match_id
                    )
                for team in teams:
                    for player in teams[team]:
                        data.append([
                            season,
                            match_id,
                            teams[team][player]['player_id'],
                            teams[team][player]['player'],
                            teams[team][player]['h_a'],
                            teams[team][player]['position'],
                            float(teams[team][player]['time']),
                            float(teams[team][player]['goals']),
                            float(teams[team][player]['assists']),
                            float(teams[team][player]['shots']),
                            float(teams[team][player]['yellow_card']),
                            float(teams[team][player]['red_card']),
                            float(teams[team][player]['xG']),
                            float(teams[team][player]['xA']),
                            3*float(teams[team][player]['goals'])+float(teams[team][player]['assists'])-0.5*float(teams[team][player]['yellow_card'])-0.5*float(teams[team][player]['red_card']),
                            3*float(teams[team][player]['xG'])+float(teams[team][player]['xA'])-0.5*float(teams[team][player]['yellow_card'])-0.5*float(teams[team][player]['red_card']),
                        ])
            df = pd.DataFrame(data=data, columns=[
                'season',
                'matchId',
                'playerId',
                'player',
                'homeOrAway',
                'position',
                'time',
                'goals',
                'assists',
                'shots',
                'yellowCard',
                'redCard',
                'xG',
                'xA',
                'bonus',
                'xB'
                ])
            df.to_csv('resources/match_players/{}/{}.csv'.format(season, season))

def get_grouped_players(seasons):
    for season in seasons:
        df = pd.read_csv('resources/match_players/{}/{}.csv'.format(season, season), index_col=0)
        grouped = df.groupby('player')
        for player, group in grouped:
            group.to_csv('resources/match_players/{}/{}.csv'.format(season, player))

def aggregate_previous_match_players(seasons):
    for season in seasons:
        df = pd.read_csv('resources/match_players/{}/{}.csv'.format(season, season), index_col=0)
        players = df['player'].unique().tolist()
        for player in players:
            df = pd.read_csv('resources/match_players/{}/{}.csv'.format(season, player), index_col=0)

            goals = 0
            assists = 0
            shots = 0
            yellowCard = 0
            redCard = 0
            xG = 0
            xA = 0
            bonus = 0
            xB = 0
            time = 0
            last_bonus = 0

            for row in df.iterrows():
                df.loc[row[0], 'goals'] = goals
                df.loc[row[0], 'assists'] = assists
                df.loc[row[0], 'shots'] = shots
                df.loc[row[0], 'yellowCard'] = yellowCard
                df.loc[row[0], 'redCard'] = redCard
                df.loc[row[0], 'xG'] = xG
                df.loc[row[0], 'xA'] = xA
                df.loc[row[0], 'bonus'] = bonus
                df.loc[row[0], 'xB'] = xB
                df.loc[row[0], 'time'] = time
                df.loc[row[0], 'lastBonus'] = last_bonus

                goals += row[1]['goals']
                assists += row[1]['assists']
                shots += row[1]['shots']
                yellowCard += row[1]['yellowCard']
                redCard += row[1]['redCard']
                xG += row[1]['xG']
                xA += row[1]['xA']
                bonus += row[1]['bonus']
                xB += row[1]['xB']
                time += row[1]['time']
                last_bonus = row[1]['bonus']

            df.to_csv('resources/match_players/{}/{}.csv'.format(season, player))

def get_complete_players(seasons):
    complete = None
    for season in seasons:
        df = pd.read_csv('resources/match_players/{}/{}.csv'.format(season, season), index_col=0)
        players = df['player'].unique().tolist()
        semi_complete = None
        for player in players:
            if semi_complete is None:
                semi_complete = pd.read_csv('resources/match_players/{}/{}.csv'.format(season, player), index_col=0)
            else:
                semi_complete = semi_complete.append(
                    pd.read_csv('resources/match_players/{}/{}.csv'.format(season, player), index_col=0),
                    ignore_index=True)
        if complete is None:
            complete = semi_complete
        else:
            complete = complete.append(semi_complete, ignore_index=True)
    complete.to_csv('resources/match_players/complete.csv')

def get_complete_labels(seasons):
    complete = None
    for season in seasons:
        if complete is None:
            complete = pd.read_csv(
                'resources/match_players/{}/{}.csv'.format(season, season),
                usecols=['season', 'matchId', 'player', 'bonus']
            )
        else:
            complete = complete.append(pd.read_csv(
                'resources/match_players/{}/{}.csv'.format(season, season),
                usecols=['season', 'matchId', 'player', 'bonus']
            ), ignore_index=True)
    complete = complete.rename(columns={'bonus': 'target'})
    complete.to_csv('resources/match_players/labels.csv')

def aggregate_match_odds(seasons):
    odds = None
    for season in seasons:
        df = pd.read_csv('resources/match_odds/{}.csv'.format(season))
        df['season'] = season

        if season in ['2019', '2020', '2021']:

            df = df[[
                'season',
                'HomeTeam',
                'AwayTeam',
                'B365H',
                'B365D',
                'B365A',
                'B365>2.5',
                'B365<2.5'
            ]]

            df = df.rename(columns={
                'B365H': 'BbAvH',
                'B365D': 'BbAvD',
                'B365A': 'BbAvA',
                'B365>2.5': 'BbAv>2.5',
                'B365<2.5': 'BbAv<2.5'
            })

        else:

            df = df[[
                'season',
                'HomeTeam',
                'AwayTeam',
                'BbAvH',
                'BbAvD',
                'BbAvA',
                'BbAv>2.5',
                'BbAv<2.5'
            ]]

        if odds is None:
            odds = df
        else:
            odds = odds.append(df, ignore_index=True)

    odds = odds.rename(columns={
        'HomeTeam': 'homeTeam',
        'AwayTeam': 'awayTeam',
        'BbAvH': 'homeWinOdds',
        'BbAvD': 'drawOdds',
        'BbAvA': 'awayWinOdds',
        'BbAv>2.5': '>2.5Odds',
        'BbAv<2.5': '<2.5Odds'
    })
    odds.to_csv('resources/match_odds/odds.csv')


def create_dataset():
    matches = pd.read_csv('resources/matches/complete.csv', index_col=0)
    players = pd.read_csv('resources/match_players/complete.csv', index_col=0)
    labels = pd.read_csv('resources/match_players/labels.csv', index_col=0)
    odds = pd.read_csv('resources/match_odds/odds.csv', index_col=0)
    df = matches.merge(players, on=['season', 'matchId'], suffixes=('', 'Player'))
    df = df[((df.homeOrAway == 'h') & (df.team == df.homeTeam)) | ((df.homeOrAway == 'a') & (df.team == df.awayTeam))]
    df = df.merge(labels, on=['season', 'matchId', 'player'])
    df = df.merge(odds, on=['season', 'homeTeam', 'awayTeam'])
    df.to_csv('resources/dataset.csv')

def get_data(seasons):
    routine(get_matches(seasons))
    get_grouped_matches(seasons)
    aggregate_previous_matches(seasons)
    get_complete_matches(seasons)
    routine(get_players(seasons))
    get_grouped_players(seasons)
    aggregate_previous_match_players(seasons)
    get_complete_players(seasons)
    get_complete_labels(seasons)
    aggregate_match_odds(seasons)
    create_dataset()

