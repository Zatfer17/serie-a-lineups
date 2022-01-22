import requests
import pandas as pd

from bs4 import BeautifulSoup

def remap_player(player):
    mapping = {
        'Stephane Wilfried Singo': 'Wilfried Stephane Singo',
        'Destiny Udogie': 'Iyenoma Destiny Udogie',
        'Matias Vina': 'Matias Viña',
        'Anderson Felipe': 'Felipe Anderson',
        'Nicolò Zaniolo': 'Nicolo Zaniolo',
        'Alvaro Morata': 'Álvaro Morata',
        'Ledesma Rodriguez Pedro': 'Pedro',
        'Diego Godin': 'Diego Godín',
        'Theo Hernandez': 'Theo Hernández',
        'Peter Stojanovic': 'Petar Stojanovic',
        'Marten Roon De': 'Marten de Roon',
        'Lobo Sandro Alex': 'Alex Sandro',
        'Matthijs Ligt De': 'Matthijs de Ligt',
        'Davide Faraoni': 'Marco Faraoni',
        'Alvaro Odriozola': 'Álvaro Odriozola',
        'Jens Larsen Stryger': 'Jens Stryger Larsen',
        'Ruslan Malinovskyi': 'Ruslan Malinovskiy',
        'Nahitan Nandez': 'Nahitan Nández',
        'Oliver Giroud': 'Olivier Giroud',
        'Pepe Reina': 'José Reina',
        'Lorenzo Silvestri De': 'Lorenzo De Silvestri',
        'Konstantinos Manolas': 'Kostas Manolas',
        'Aleksej Miranchuk': 'Aleksey Miranchuk',
        'Duvan Zapata': 'Duván Zapata',
        'Arnor Sigurdsson': 'Arnór Sigurdsson',
        ' Hamed Traorè': 'Hamed Junior Traore',
        'Galvao Pedro Joao': 'João Pedro',
        'Rafael Leao': 'Rafael Leão',
        'Nwankwo Tochukwu Simeon': 'Simy',
        'Jaime Cuadrado': 'Juan Cuadrado',
        'Giovanni Lorenzo Di': 'Giovanni Di Lorenzo',
        'Leonardo Spinazzola': 'Leonardo Spinazzola',
        'Maria Josè Callejon': 'José Callejón',
        'Franck Kessiè': 'Franck Kessié',
        'Alconchel Alberto Luis': 'Luis Alberto',
        'Nicolas Gonzalez': 'Nicolás González',
        'Lautaro Martinez': 'Lautaro Martínez',
        "M'Bala Nzola": 'M&#039;Bala Nzola',
        'Silva da Luiz Danilo': 'Danilo',
        'Stefan Vrij De': 'Stefan de Vrij',
        'Joaquin Correa': 'Joaquín Correa',
        'Patricio Rui': 'Rui Patrício',
        'Berat Djimsiti': 'Berat Gjimshiti',
        'Roger Ibanez': 'Ibañez',
        'Ramos Felipe Luiz': 'Luiz Felipe',
        'Duarte Rui Mario': 'Mário Rui',
        'Fabian Ruiz': 'Fabián',
        'Frank Ribery': 'Franck Ribéry'
    }
    
    if player in mapping:
        return mapping[player]
    else:
        return player
    
def remap_team(team):
    mapping = {
        'Milan': 'AC Milan'
    }
    
    if team in mapping:
        return mapping[team]
    else:
        return team

def get_league_teams(league_name):
    
    url = 'https://leghe.fantacalcio.it/{}/area-gioco/rose'.format(league_name)
    
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    
    league_data = []

    team_names = soup.find_all(class_='media-heading')
    rows = soup.find_all(class_='smart-table table-striped fixed table no-margin has-subheader')
    for row, team_name in zip(rows, team_names):
        team_name = team_name.text

        roles = row.find_all(class_='cell-text cell-role cell-primary x1 smart-x2 mantra-x3 free-player-hidden')
        players = row.find_all(class_='player-link')

        for role, player in zip(roles, players):
            role = role.text

            name_page = requests.get(player['href'])
            soup_name = BeautifulSoup(name_page.text, 'html.parser')

            box = soup_name.find(class_='stickem-container')
            name = box.find(class_='img-responsive')['title'].split(' ')
            name.reverse()
            name = ' '.join(name)
            image = box.find(class_='img-responsive')['src']
            squad = box.find_all(class_='col-lg-6 col-md-6 col-sm-12 col-xs-12')[4].text.split(' ')[1]

            league_data.append([team_name, role, name, squad, image])

    league = pd.DataFrame(data=league_data, columns=['fanta_team', 'role', 'player', 'team', 'image'])
    league['player'] = league['player'].apply(lambda x: remap_player(x))
    league['team'] = league['team'].apply(lambda x: remap_team(x))
    league.to_csv('data/leghe_fantacalcio/leagues/{}.csv'.format(league_name), index=False)
    league.head()