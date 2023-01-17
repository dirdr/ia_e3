import numpy as np
from random_n import pvp_multiple_match, get_score_pretty_string

def main_pvp() -> None:
    """
    main function for ia vs ia matchs
    """
    number_of_game: int = 10
    score_100_100: np.ndarray = pvp_multiple_match(
        number_of_game, nog_player_0=100, nog_player_1=100
    )
    score_100_1000: np.ndarray = pvp_multiple_match(
        number_of_game, nog_player_0=100, nog_player_1=1000
    )
    score_100_10000: np.ndarray = pvp_multiple_match(
        number_of_game, nog_player_0=1000, nog_player_1=10000, p=True
    )
    print(score_100_100)
    print(get_score_pretty_string(score_100_100, number_of_game))
    print(get_score_pretty_string(score_100_1000, number_of_game))
    print(get_score_pretty_string(score_100_10000, number_of_game))


def main() -> None:
    main_pvp()


if __name__ == "__main__":
    main()
