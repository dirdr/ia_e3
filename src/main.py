import numpy as np
from aiproject.battle.AivsAi import MonteCarloPlayer, MonteCarloTreeSearchPlayer
from aiproject.battle.battle import Player, Battle

def main() -> None:
    monte_carlo = MonteCarloPlayer()
    monte_carlo_search = MonteCarloTreeSearchPlayer()
    battle = Battle(monte_carlo, monte_carlo)

if __name__ == "__main__":
    main()
