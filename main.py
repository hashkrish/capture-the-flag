import random
import time
import os


def clear_screen():
    if os.name == "nt":  # for Windows
        os.system("cls")
    else:  # for Unix/Linux/Mac
        os.system("clear")


# Initialize game variables
# territory_size = 10
# num_players_per_team = 2
# num_teams = 2
# flag_locations = [0, territory_size - 1]
# players = []
# team_names = ["A", "B"]


class Game:
    def __init__(
        self,
        territory_size: int,
        num_players_per_team: int,
        num_teams: int,
        flag_locations: list,
        players: list,
        team_names: list,
    ):
        self.territory_size = territory_size
        self.num_players_per_team = num_players_per_team
        self.num_teams = num_teams
        self.flag_locations = flag_locations
        self.players = players
        self.team_names = team_names

        # Create players
        for team in range(1, num_teams + 1):
            for player_num in range(num_players_per_team):
                player_name = f"{team_names[team-1]}{player_num + 1}"
                player_position = random.randint(0, territory_size - 1)
                players.append(Player(player_name, team, player_position))

    def __str__(self):
        return f"Game with {self.num_teams} teams, {self.num_players_per_team} players per team, and a territory size of {self.territory_size}."

    def __repr__(self):
        return f"Game({self.territory_size}, {self.num_players_per_team}, {self.num_teams}, {self.flag_locations}, {self.players}, {self.team_names})"

    def run(self):
        # Main game loop
        while True:
            positions = {t: [] for t in range(self.territory_size)}
            for i, flag_location in enumerate(self.flag_locations):
                positions[flag_location].append(self.team_names[i] + "*")
            for player in self.players:
                positions[player.position].append(player)

            clear_screen()
            print("\nCurrent player positions:")
            for position, players_ in positions.items():
                print(f"{position}: {[ p for p in players_ ]}")

            # Check for flag capture
            for team in range(1, self.num_teams + 1):
                flag_position = self.flag_locations[team - 1]
                players_in_team = [
                    player for player in self.players if player.team == team
                ]

                if any(player.position == flag_position for player in players_in_team):
                    print(f"\nTeam {team} captured the flag!")
                    exit()

            # Move players
            for player in self.players:
                direction = random.choice([-1, 1])
                player.move(direction)

            time.sleep(1)


# Define Player class
class Player:
    def __init__(self, name, team, position):
        self.name = name
        self.team = team
        self.position = position

    def move(self, direction):
        new_position = self.position + direction
        if 0 <= new_position < game.territory_size:
            self.position = new_position

    def __str__(self):
        return f"{self.name} (Team {self.team}) at position {self.position}"

    def __repr__(self):
        return f"{self.name}"


class Flag:
    def __init__(self, name, team, position):
        self.team = team
        self.position = position

    def __str__(self):
        return f"(Team {self.team} Flag) at position {self.position}"

    def __repr__(self):
        return f"{self.team}*"

    def __eq__(self, other):
        if isinstance(other, int):
            return self.position == other
        return self.__repr__() == other.__repr__()


# Create Game
game = Game(
    territory_size=10,
    num_players_per_team=2,
    num_teams=2,
    flag_locations=[0, 9],
    players=[],
    team_names=["A", "B"],
)
game.run()
